from algs.DQN.dqn_agent import DQNAgent
from algs.DRQN.drqn_agent import DRQNAgent
from algs.FIXTIME.fixtime_agent import FIXTIMEAgent
from algs.FRAP.frap_agent import FRAPAgent
from algs.FRAPPlus.frapplus_agent import FRAPPlusAgent
from algs.FRAPRQ.fraprq_agent import FRAPRQAgent
from algs.MAXPRESSURE.maxpressure_agent import MAXPRESSUREAgent
from algs.TDDD.tddd_agent import TDDDAgent
from algs.MetaLight.metalight_agent import MetaLightAgent
from algs.SOTL.sotl_agent import SOTLAgent
from algs.WEBSTER.webster_agent import WEBSTERAgent
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
    # "stop_vehicle_thres1",
    # "stop_vehicle_thres1",
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
    "stop_vehicle_thres1",
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
    "sum_stop_vehicle_thres1": -0.25
}

DIC_AGENT_CONF_SOTL = {
    "PHI_MIN": 0,
    "THETA": 10,
    "MU": 5,
}

DIC_AGENT_CONF_WEBSTER = {
    "L_LANE": 3,
    "K1": 1.5,
    "K2": 5,
    "Y_MAX": 1650,
}
DIC_AGENT_CONF_FIXTIME = {
    "TIME_PHASE_0": 15,
    "TIME_PHASE_1": 15,
    "TIME_PHASE_2": 15,
    "TIME_PHASE_3": 15,
    "TIME_PHASE_4": 15,
    "TIME_PHASE_5": 15,
    "TIME_PHASE_6": 15,
    "TIME_PHASE_7": 15,
}
DIC_AGENT_CONF_MAXPRESSURE = {
    "G_MIN": 5,
}
DIC_AGENT_CONF_TDDD = {
    "LR": 0.001,
    "LR_ACTOR" : 0.0001,
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 32,
    "EPOCHS": 100,

    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,

    "NORMAL_FACTOR": 20,
    "POLICY_FREQ": 2,
    "TAU": 0.2,

    "EXPL_NOISE": 0.2,
    "EXPL_NOISE_END": 0.1,
    "EXPL_NOISE_DECAY": 0.98,
    "EXPL_NOISE_MIN": -0.1,
    "EXPL_NOISE_MAX": 0.3,

    "POLICY_NOISE": 0.05,
    "POLICY_NOISE_MIN": 0,
    "POLICY_NOISE_MAX": 0.5,

    "ENABLE_CONTEXT": True,
    "HISTORY_LENGTH": 20,
    "HIDDEN_DIM": 10,
    "BETA_CLIP": 1.5,
    "ENABLE_ADAPT": True,
}


DIC_AGENT_CONF_DRQN = {
    "LR": 0.001,
    "LR_DECAY": 1,
    "MIN_LR": 0.0001,
    "SAMPLE_SIZE": 5000,
    "BATCH_SIZE": 32,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 50000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "N_LAYER": 2,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.90,
    "MIN_EPSILON": 0.2,
    "NORMAL_FACTOR": 20,

    "HISTORY_LEN": 20,
    "HIDDEN_DIM": 10,
}

DIC_AGENT_CONF_METALIGHT = {
    "LR": 0.001,
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

    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 5000,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "NORMAL_FACTOR": 20,
    "EARLY_STOP": False,
}

DIC_AGENT_CONF_FRAPPLUS = {
    "LR": 0.001,
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 32,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "NORMAL_FACTOR": 20,
}
DIC_AGENT_CONF_FRAP = {
    "LR": 0.001,
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 32,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "NORMAL_FACTOR": 20,
}
DIC_AGENT_CONF_FRAPRQ = {
    "LR": 0.001,
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 32,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "NORMAL_FACTOR": 20,
    "HIDDEN_DIM": 5,
    "HISTORY_LEN": 20,
}
DIC_AGENT_CONF_DQN = {
    "LR": 0.001,
    "LR_DECAY": 1,
    "MIN_LR": 0.0001,
    "SAMPLE_SIZE": 5000,
    "BATCH_SIZE": 32,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 50000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "N_LAYER": 2,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.90,
    "MIN_EPSILON": 0.2,
    "NORMAL_FACTOR": 20,
}

DIC_AGENTS = {
    "MetaLight": MetaLightAgent,
    "FRAPPlus": FRAPPlusAgent,
    "DQN": DQNAgent,
    "SOTL": SOTLAgent,
    "FRAP": FRAPAgent,
    "TDDD": TDDDAgent,
    "DRQN": DRQNAgent,
    "WEBSTER": WEBSTERAgent,
    "FIXTIME": FIXTIMEAgent,
    "MAXPRESSURE": MAXPRESSUREAgent,
    "FRAPRQ": FRAPRQAgent,
}
RL_ALGORITHM = [
    "MetaLight",
    "FRAPPlus",
    "DQN",
    "FRAP",
    "TDDD",
    "DRQN",
    "FRAPRQ",
]
TRAD_ALGORITHM = [
    "SOTL",
    "WEBSTER",
    "FIXTIME",
    "MAXPRESSURE",
]
DIC_ENVS = {
    # "sumo": SumoEnv,
    "AnonEnv": AnonEnv,
    # 'dist':
}
