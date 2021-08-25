from algs.FRAP.frap_agent import FRAPAgent
from algs.FRAPPlus.frapplus_agent import FRAPPlusAgent
from envs.anon_env import AnonEnv

# from envs.sumo_env import SumoEnv

DIC_FEATURE_DIM = dict(
    cur_phase=(8,),
    lane_num_vehicle=(8,),
    # "time_this_phase",
    # "vehicle_position_img",
    # "vehicle_speed_img",
    # "vehicle_acceleration_img",
    # "vehicle_waiting_time_img",
    # "lane_num_vehicle",
    # "lane_num_vehicle_been_stopped_thres01",
    # "lane_num_vehicle_been_stopped_thres1",
    # "lane_queue_length",
    # "lane_num_vehicle_left",
    # "lane_sum_duration_vehicle_left",
    # "lane_sum_waiting_time",
    # "terminal"
)

LIST_STATE_FEATURE = [
    "cur_phase",
    "cur_phase_index",
    "time_this_phase",
    "vehicle_position_img",
    "vehicle_speed_img",
    "vehicle_acceleration_img",
    "vehicle_waiting_time_img",
    "lane_num_vehicle",
    "lane_num_vehicle_been_stopped_thres01",
    "lane_num_vehicle_been_stopped_thres1",
    "lane_queue_length",
    "lane_num_vehicle_left",
    "lane_sum_duration_vehicle_left",
    "lane_sum_waiting_time",
    "terminal"
]

DIC_REWARD_INFO = {
    "flickering": 0,
    "sum_lane_queue_length": 0,
    "sum_lane_wait_time": 0,
    "sum_lane_num_vehicle_left": 0,
    "sum_duration_vehicle_left": 0,
    "sum_num_vehicle_been_stopped_thres01": 0,
    "sum_num_vehicle_been_stopped_thres1": -0.25
}

DIC_SOTL_AGENT_CONF = {
    "PHI": 5,
    "MIN_GREEN_VEC": 3,
    "MAX_RED_VEC": 6,
}

DIC_METALIGHT_AGENT_CONF = {
    "LEARNING_RATE": 0.001,
    "ALPHA": 0.1,
    "MIN_ALPHA": 0.00025,
    "ALPHA_DECAY_RATE": 0.95,
    "ALPHA_DECAY_STEP": 100,
    "BETA": 0.1,
    "LR_DECAY": 1,
    "MIN_LR": 0.0001,
    "SAMPLE_SIZE": 30,
    'UPDATE_START': 100,
    'UPDATE_PERIOD': 10,
    "TEST_PERIOD": 50,
    "BATCH_SIZE": 32,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 5000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "N_LAYER": 2,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
    "SEPARATE_MEMORY": False,
    "NORMAL_FACTOR": 20,
    "TRAFFIC_FILE": "cross.2phases_rou01_equal_450.xml",
    "EARLY_STOP_LOSS": "val_loss",
    "DROPOUT_RATE": 0,
    "MORE_EXPLORATION": False
}

DIC_FRAPPLUS_AGENT_CONF = {
    "LEARNING_RATE": 0.001,
    "ALPHA": 0.1,
    "MIN_ALPHA": 0.00025,
    "ALPHA_DECAY_RATE": 0.95,
    "ALPHA_DECAY_STEP": 100,
    "BETA": 0.1,
    "LR_DECAY": 1,
    "MIN_LR": 0.0001,
    "SAMPLE_SIZE": 1000,
    'UPDATE_START': 100,
    'UPDATE_PERIOD': 10,
    "TEST_PERIOD": 50,
    "BATCH_SIZE": 32,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "N_LAYER": 2,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
    "SEPARATE_MEMORY": False,
    "NORMAL_FACTOR": 20,
    "TRAFFIC_FILE": "None",
    "EARLY_STOP_LOSS": "val_loss",
    "DROPOUT_RATE": 0,
    "MORE_EXPLORATION": False
}
DIC_FRAP_AGENT_CONF = {
    "LEARNING_RATE": 0.001,
    "LR_DECAY": 1,
    "MIN_LR": 0.0001,
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 32,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "N_LAYER": 2,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "NORMAL_FACTOR": 20,
}
DIC_AGENTS = {
    # "MetaLight": MetaLightAgent,
    "FRAPPlus": FRAPPlusAgent,
    # "SOTL": SOTLAgent,
    "FRAP": FRAPAgent,
}

DIC_ENVS = {
    # "sumo": SumoEnv,
    "AnonEnv": AnonEnv,
    # 'dist':
}

