from keras.models import model_from_json, load_model
import random
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from algs.agent import Agent
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.merge import concatenate, add, multiply
from keras import backend as K
import numpy as np


def slice_tensor(x, index):
    x_shape = K.int_shape(x)
    if len(x_shape) == 3:
        return x[:, index, :]
    elif len(x_shape) == 2:
        return Reshape((1,))(x[:, index])


def keras_relation(x, dic_traffic_env_conf):
    relations = dic_traffic_env_conf["LANE_PHASE_INFO"]['relation']
    num_phase = len(relations)
    relations = np.array(relations).reshape((1, num_phase, num_phase - 1))
    batch_size = K.shape(x)[0]
    constant = K.constant(relations)
    constant = K.tile(constant, (batch_size, 1, 1))
    return constant


class Selector(Layer):
    def __init__(self, select, d_phase_encoding, d_action, **kwargs):
        super(Selector, self).__init__(**kwargs)
        self.select = select
        self.d_phase_encoding = d_phase_encoding
        self.d_action = d_action
        self.select_neuron = K.constant(value=self.select,
                                        shape=(1, self.d_phase_encoding))

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Selector, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        batch_size = K.shape(x)[0]
        constant = K.tile(self.select_neuron, (batch_size, 1))
        return K.min(K.cast(K.equal(x, constant), dtype="float32"), axis=-1,
                     keepdims=True)

    def get_config(self):
        config = {"select": self.select,
                  "d_phase_encoding": self.d_phase_encoding,
                  "d_action": self.d_action}
        base_config = super(Selector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return batch_size, self.d_action


class FRAPAgent(Agent):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path,
                 round_number):
        super().__init__(dic_agent_conf, dic_traffic_env_conf, dic_path,
                         round_number)
        self.decay_epsilon(self.round_number)

        session = tf.Session()
        KTF.set_session(session)

        self.lane_phase_info = dic_traffic_env_conf["LANE_PHASE_INFO"]
        self.list_lane = self.lane_phase_info['start_lane']
        self.list_phase = self.lane_phase_info['phase']
        self.num_phase = len(self.list_phase)

        if self.round_number == 0:
            self.q_network = self.build_network()
            self.q_network_bar = self.build_network_bar()
        else:
            self.load_network("round_%d" % (self.round_number - 1))
            q_bar_freq = self.dic_agent_conf["UPDATE_Q_BAR_FREQ"]
            q_bar_number = (self.round_number - 1) // q_bar_freq * q_bar_freq
            self.load_network_bar("round_%d" % q_bar_number)

    def build_network(self):
        dic_input_node = {}
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            dim_feature = self.dic_traffic_env_conf[
                "DIC_FEATURE_DIM"][feature_name]
            dic_input_node[feature_name] = Input(shape=dim_feature,
                                                 name="input_" + feature_name)
        p = Activation('sigmoid')(Embedding(2, 4)(dic_input_node["cur_phase"]))

        LANE_ORDER = self.dic_traffic_env_conf["LANE_PHASE_INFO"][
            'start_lane']
        d = Dense(4, activation="sigmoid", name="num_vec_mapping")
        dic_lane = {}
        for i, m in enumerate(LANE_ORDER):
            tmp_vec = Lambda(slice_tensor, arguments={"index": i},
                             name="vec_%d" % i)(
                dic_input_node["lane_num_vehicle"])
            tmp_vec = d(tmp_vec)
            tmp_phase = Lambda(slice_tensor,
                               arguments={"index": i},
                               name="phase_%d" % i)(p)
            dic_lane[m] = concatenate([tmp_vec, tmp_phase],
                                      name="lane_%d" % i)

        list_phase_pressure = []
        lane_embedding = Dense(16, activation="relu", name="lane_embedding")

        DICT_LANE_PHASE = self.dic_traffic_env_conf[
            "LANE_PHASE_INFO"]['phase_startLane_mapping']
        LANE_PHASE = [DICT_LANE_PHASE[i] for i in
                      sorted(DICT_LANE_PHASE.keys())]

        for lane_combine in LANE_PHASE:
            lane1, lane2 = lane_combine[0], lane_combine[1]
            list_phase_pressure.append(add([lane_embedding(dic_lane[lane1]),
                                            lane_embedding(dic_lane[lane2])]))

        constant_layer = \
            Lambda(keras_relation, arguments={
                "dic_traffic_env_conf": self.dic_traffic_env_conf},
                   name="constant")(dic_input_node["lane_num_vehicle"])
        relation_embedding = Embedding(
            2, 4, name="relation_embedding")(constant_layer)

        list_phase_pressure_recomb = []

        for i in range(self.num_phase):
            for j in range(self.num_phase):
                if i != j:
                    list_phase_pressure_recomb.append(
                        concatenate(
                            [list_phase_pressure[i], list_phase_pressure[j]],
                            name="concat_compete_phase_%d_%d" % (i, j)))

        list_phase_pressure_recomb = concatenate(list_phase_pressure_recomb,
                                                 name="concat_all")
        feature_map = Reshape((self.num_phase, self.num_phase - 1, 32)
                              )(list_phase_pressure_recomb)
        lane_conv = Conv2D(20,
                           kernel_size=(1, 1), activation="relu",
                           name="lane_conv")(feature_map)
        relation_conv = Conv2D(20, kernel_size=(1, 1), activation="relu",
                               name="relation_conv")(relation_embedding)
        combine_feature = multiply([lane_conv, relation_conv],
                                   name="combine_feature")
        hidden_layer = Conv2D(20, kernel_size=(1, 1), activation="relu",
                              name="combine_conv")(combine_feature)
        before_merge = Conv2D(1, kernel_size=(1, 1), activation="linear",
                              name="before_merge")(hidden_layer)
        q_values = Lambda(lambda x: K.sum(x, axis=2), name="q_values")(
            Reshape((self.num_phase, self.num_phase - 1))(before_merge))
        input = [dic_input_node[feature_name] for feature_name in
                  sorted(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])]
        network = Model(input=input, outputs=q_values)

        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LR"],
                                       epsilon=1e-08),
                        loss='mean_squared_error')
        return network

    def load_network(self, file_name):
        file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network = load_model(
            os.path.join(file_path, "%s.h5" % file_name),
            custom_objects={"Selector": Selector})
        print("succeed in loading model %s" % file_name)

    def load_network_bar(self, file_name):
        file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network_bar = load_model(
            os.path.join(file_path, "%s.h5" % file_name),
            custom_objects={"Selector": Selector})
        print("succeed in loading model %s" % file_name)

    def save_network(self, file_name):
        self.q_network.save(
            os.path.join(self.dic_path["PATH_TO_MODEL"],
                         "%s.h5" % file_name))

    def build_network_bar(self):
        network_structure = self.q_network.to_json()
        network_weights = self.q_network.get_weights()
        network = model_from_json(network_structure,
                                  custom_objects={"Selector": Selector})
        network.set_weights(network_weights)
        network.compile(optimizer=Adam(), loss='mean_squared_error')
        return network

    def prepare_Xs_Y(self, sample_set):
        Xs = [[] for _ in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]
        Y = []
        for i in range(len(sample_set)):
            state, action, next_state, reward, instant_reward, _ = \
                sample_set[i]
            _state = []
            _next_state = []
            for i, feature_name in \
                    enumerate(self.dic_traffic_env_conf["LIST_STATE_FEATURE"]):
                _state.append([state[feature_name]])
                _next_state.append([next_state[feature_name]])
                Xs[i].append(state[feature_name])
            target = self.q_network.predict(_state)
            next_state_qvalues = self.q_network_bar.predict(_next_state)

            final_target = np.copy(target[0])
            final_target[action] = \
                reward / self.dic_agent_conf["NORMAL_FACTOR"] \
                + self.dic_agent_conf["GAMMA"] * np.max(next_state_qvalues[0])

            Y.append(final_target)
        self.Xs = Xs
        self.Y = np.array(Y)

    def convert_state_to_input(self, s):
        input = []
        dic_phase_expansion = self.dic_traffic_env_conf[
            "LANE_PHASE_INFO"]['phase_map']
        for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if feature == "cur_phase":
                input.append(
                    np.array([dic_phase_expansion[s[feature][0]]]))
            else:
                input.append(np.array([s[feature]]))
        return input

    def choose_action(self, state, choice_random=True):
        input = self.convert_state_to_input(state)
        q_values = self.q_network.predict(input)
        if random.random() <= self.dic_agent_conf["EPSILON"] and choice_random:
            action = random.randrange(len(q_values[0]))
        else:
            action = np.argmax(q_values[0])
        return action

    def train_network(self):
        epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))
        self.q_network.fit(self.Xs, self.Y,
                           batch_size=len(self.Y),
                           epochs=epochs, verbose=0)
