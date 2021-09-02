from algs.DQN.dqn_agent import DQNAgent
from algs.FRAP.frap_agent import FRAPAgent
from algs.FRAPPlus.frapplus_agent import FRAPPlusAgent
from algs.MetaLight.metalight_agent import MetaLightAgent
from algs.SOTL.sotl_agent import SOTLAgent
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
    "stop_vehicle_thres01",
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
    "sum_stop_vehicle_thres01": 0,
    "sum_stop_vehicle_thres1": -0.25
}

DIC_AGENT_CONF_SOTL = {
    "PHI_MIN": 0,
    "THETA": 10,
    "MU": 5,
}

DIC_AGENT_CONF_METALIGHT = {
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

    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 5000,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "NORMAL_FACTOR": 20,
    "EARLY_STOP": False,
    "TASK_COUNT": 3,
}

DIC_AGENT_CONF_FRAPPLUS = {
    "LEARNING_RATE": 0.001,
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
    "LEARNING_RATE": 0.001,
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

DIC_AGENT_CONF_DQN = {
    "LEARNING_RATE": 0.001,
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
}
RL_ALGORITHM = [
    "MetaLight",
    "FRAPPlus",
    "DQN",
    "FRAP",
]
TRAD_ALGORITHM = [
    "SOTL",
]
DIC_ENVS = {
    # "sumo": SumoEnv,
    "AnonEnv": AnonEnv,
    # 'dist':
}
