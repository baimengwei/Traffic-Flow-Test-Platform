LIST_STATE_FEATURE = [
    "cur_phase_index",
    "lane_vehicle_cnt",

    "time_this_phase",
    "stop_vehicle_thres1",
    "lane_queue_length",
    "lane_vehicle_left_cnt",
    "lane_waiting_time",
    "terminal"
]

DIC_REWARD_INFO = {
    "sum_lane_queue_length": 0,
    "sum_lane_wait_time": 0,
    "sum_lane_vehicle_left_cnt": 0,
    "sum_duration_vehicle_left": 0,
    "sum_stop_vehicle_thres1": -0.25
}

RL_ALGORITHM = [
    "MetaLight",
    "FRAPPlus",
    "DQN",
    "FRAP",
    "TDDD",
    "DRQN",
    "FRAPRQ",
    "MetaDQN",
    "MetaDQNAdapt",
]

TRAD_ALGORITHM = [
    "SOTL",
    "WEBSTER",
    "FIXTIME",
    "MAXPRESSURE",
]
