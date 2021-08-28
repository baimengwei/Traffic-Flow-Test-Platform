dic_path = {
    "PATH_TO_MODEL_ROOT": "./records/weights/FRAP_TEST",
    "PATH_TO_WORK_ROOT": "./records/workspace/FRAP_TEST",
    "PATH_TO_ERROR_ROOT": "./records/errors/FRAP_TEST",
    "PATH_TO_GRADIENT_ROOT": "./records/gradient/FRAP_TEST",
    "PATH_TO_FIGURE_ROOT": "./records/figures/FRAP_TEST",
    "PATH_TO_DATA_ROOT": "./data/scenario/",
    "PATH_TO_MODEL": "./records/weights/FRAP_TEST/TEST_WORK_PLACE",
    "PATH_TO_WORK": "./records/workspace/FRAP_TEST/TEST_WORK_PLACE",
    "PATH_TO_GRADIENT": "./records/gradient/FRAP_TEST/TEST_WORK_PLACE",
    "PATH_TO_ERROR": "./records/errors/FRAP_TEST/TEST_WORK_PLACE",
    "PATH_TO_FIGURE": "./records/figures/FRAP_TEST/TEST_WORK_PLACE",
    "PATH_TO_DATA": "./data/scenario/demo_train_1364",
    "PATH_TO_LOG": "./records/workspace/FRAP_TEST/TEST_WORK_PLACE"
                   "/train_round",
    "PATH_TO_PRETRAIN_MODEL_ROOT": "./records/weights/default/",
    "PATH_TO_PRETRAIN_WORK_ROOT": "./records/workspace/default/"
}
dic_traffic_env = {
    "SAVEREPLAY": False,
    "EPISODE_LEN": 3600,
    "DONE_ENABLE": False,
    "REWARD_NORM": False,
    "ENV_DEBUG": False,
    "FAST_BATCH_SIZE": 3,
    "MODEL_NAME": "FRAP",
    "ENV_NAME": "AnonEnv",
    "TRAFFIC_FILE": "demo_train_1364",
    "DIC_FEATURE_DIM": {"cur_phase": [8], "lane_num_vehicle": [8]},
    "LIST_STATE_FEATURE_ALL": [
        "cur_phase",
        "cur_phase_index",
        "time_this_phase",
        "vehicle_position_img",
        "vehicle_speed_img",
        "vehicle_acceleration_img",
        "vehicle_waiting_time_img",
        "lane_num_vehicle",
        "stop_vehicle_thres1",
        "stop_vehicle_thres1",
        "lane_queue_length",
        "lane_num_vehicle_left",
        "lane_sum_duration_vehicle_left",
        "lane_sum_waiting_time",
        "terminal"
    ],
    "DIC_REWARD_INFO_ALL": {
        "flickering": 0,
        "sum_lane_queue_length": 0,
        "sum_lane_wait_time": 0,
        "sum_lane_num_vehicle_left": 0,
        "sum_duration_vehicle_left": 0,
        "sum_num_vehicle_been_stopped_thres01": 0,
        "stop_vehicle_thres1": -0.25
    },
    "TRAFFIC_CATEGORY": {
        "train": {
            "4a": [
                "hangzhou_baochu_tiyuchang_1h_10_11_2021",
                "hangzhou_shenban_shixiang_1h_12_13_1448",
                "hangzhou_baochu_tiyuchang_1h_12_13_1573",
                "hangzhou_tianmushan_xueyuan_1h_11_12_1744"
            ],
            "4b": [
                "hangzhou_qingchun_yanan_1h_16_17_1504",
                "hangzhou_tianmushan_xueyuan_1h_10_11_2055",
                "hangzhou_shenban_shixiang_1h_11_12_1530",
                "hangzhou_tianmushan_xueyuan_1h_12_13_1664"
            ],
            "6a": [
                "hangzhou_baochu_tiyuchang_1h_13_14_1798",
                "hangzhou_qingchun_yanan_1h_18_19_1544",
                "hangzhou_qingchun_yanan_1h_17_18_1497",
                "hangzhou_baochu_tiyuchang_1h_14_15_1865"
            ],
            "6c": [
                "hangzhou_shenban_shixiang_1h_1309",
                "hangzhou_baochu_tiyuchang_1h_15_16_1931",
                "hangzhou_tianmushan_xueyuan_1h_1247",
                "hangzhou_tianmushan_xueyuan_1h_14_15_1868"
            ],
            "6e": [
                "hangzhou_qingchun_yanan_1h_7_8_1289",
                "hangzhou_shenban_shixiang_1h_17_18_1772",
                "hangzhou_qingchun_yanan_1h_11_12_1453",
                "hangzhou_tianmushan_xueyuan_1h_15_16_1957"
            ],
            "8": [
                "hangzhou_qingchun_yanan_1h_10_11_1406",
                "hangzhou_shenban_shixiang_1h_15_16_1655",
                "hangzhou_baochu_tiyuchang_1h_17_18_2108",
                "hangzhou_baochu_tiyuchang_1h_18_19_1770",
                "hangzhou_tianmushan_xueyuan_1h_18_19_1484",
                "hangzhou_tianmushan_xueyuan_1h_8_9_2159"
            ]
        },
        "valid": {
            "4a": [
                "hangzhou_qingchun_yanan_1h_15_16_1554"
            ],
            "4b": [
                "hangzhou_qingchun_yanan_1h_14_15_1534"
            ],
            "6a": [
                "hangzhou_shenban_shixiang_1h_14_15_1614"
            ],
            "6c": [
                "hangzhou_shenban_shixiang_1h_13_14_1653"
            ],
            "6e": [
                "hangzhou_shenban_shixiang_1h_18_19_1489"
            ],
            "8": [
                "hangzhou_qingchun_yanan_1h_12_13_1373"
            ]
        },
        "test": {
            "4c": [
                "hangzhou_qingchun_yanan_1h_13_14_1536"
            ],
            "4d": [
                "hangzhou_tianmushan_xueyuan_1h_9_10_2178"
            ],
            "6b": [
                "hangzhou_baochu_tiyuchang_1h_8_9_2231"
            ],
            "6d": [
                "hangzhou_shenban_shixiang_1h_8_9_2032"
            ],
            "6f": [
                "hangzhou_tianmushan_xueyuan_1h_17_18_2062"
            ]
        },
        "city": {
            "4a": [
                "real-3701022106-1h-1350",
                "ngsim_lsr_inter_1_3600_2172",
                "ngsim_lsr_inter_0_3600_1786"
            ],
            "4b": [
                "real-3701022124-1h-2255"
            ],
            "6a": [
                "real-3701055130-1h-1917",
                "ngsim_lsr_inter_3_3600_2264"
            ]
        },
        "train_all": [
            "hangzhou_baochu_tiyuchang_1h_10_11_2021",
            "hangzhou_shenban_shixiang_1h_12_13_1448",
            "hangzhou_baochu_tiyuchang_1h_12_13_1573",
            "hangzhou_tianmushan_xueyuan_1h_11_12_1744",
            "hangzhou_qingchun_yanan_1h_16_17_1504",
            "hangzhou_tianmushan_xueyuan_1h_10_11_2055",
            "hangzhou_shenban_shixiang_1h_11_12_1530",
            "hangzhou_tianmushan_xueyuan_1h_12_13_1664",
            "hangzhou_baochu_tiyuchang_1h_13_14_1798",
            "hangzhou_qingchun_yanan_1h_18_19_1544",
            "hangzhou_qingchun_yanan_1h_17_18_1497",
            "hangzhou_baochu_tiyuchang_1h_14_15_1865",
            "hangzhou_shenban_shixiang_1h_1309",
            "hangzhou_baochu_tiyuchang_1h_15_16_1931",
            "hangzhou_tianmushan_xueyuan_1h_1247",
            "hangzhou_tianmushan_xueyuan_1h_14_15_1868",
            "hangzhou_qingchun_yanan_1h_7_8_1289",
            "hangzhou_shenban_shixiang_1h_17_18_1772",
            "hangzhou_qingchun_yanan_1h_11_12_1453",
            "hangzhou_tianmushan_xueyuan_1h_15_16_1957",
            "hangzhou_qingchun_yanan_1h_10_11_1406",
            "hangzhou_shenban_shixiang_1h_15_16_1655",
            "hangzhou_baochu_tiyuchang_1h_17_18_2108",
            "hangzhou_baochu_tiyuchang_1h_18_19_1770",
            "hangzhou_tianmushan_xueyuan_1h_18_19_1484",
            "hangzhou_tianmushan_xueyuan_1h_8_9_2159"
        ],
        "valid_all": [
            "hangzhou_qingchun_yanan_1h_15_16_1554",
            "hangzhou_qingchun_yanan_1h_14_15_1534",
            "hangzhou_shenban_shixiang_1h_14_15_1614",
            "hangzhou_shenban_shixiang_1h_13_14_1653",
            "hangzhou_shenban_shixiang_1h_18_19_1489",
            "hangzhou_qingchun_yanan_1h_12_13_1373"
        ],
        "test_all": [
            "hangzhou_qingchun_yanan_1h_13_14_1536",
            "hangzhou_tianmushan_xueyuan_1h_9_10_2178",
            "hangzhou_baochu_tiyuchang_1h_8_9_2231",
            "hangzhou_shenban_shixiang_1h_8_9_2032",
            "hangzhou_tianmushan_xueyuan_1h_17_18_2062"
        ],
        "city_all": [
            "real-3701022106-1h-1350",
            "ngsim_lsr_inter_1_3600_2172",
            "ngsim_lsr_inter_0_3600_1786",
            "real-3701022124-1h-2255",
            "real-3701055130-1h-1917",
            "ngsim_lsr_inter_3_3600_2264"
        ],
        "traffic_info": {
            "hangzhou_baochu_tiyuchang_1h_10_11_2021": [
                "train",
                "4a",
                "roadnet_p4a_lt.json",
                "flow.json"
            ],
            "hangzhou_shenban_shixiang_1h_12_13_1448": [
                "train",
                "4a",
                "roadnet_p4a_lt.json",
                "flow.json"
            ],
            "hangzhou_baochu_tiyuchang_1h_12_13_1573": [
                "train",
                "4a",
                "roadnet_p4a_lt.json",
                "flow.json"
            ],
            "hangzhou_tianmushan_xueyuan_1h_11_12_1744": [
                "train",
                "4a",
                "roadnet_p4a_lt.json",
                "flow.json"
            ],
            "hangzhou_qingchun_yanan_1h_16_17_1504": [
                "train",
                "4b",
                "roadnet_p4b_lt.json",
                "flow.json"
            ],
            "hangzhou_tianmushan_xueyuan_1h_10_11_2055": [
                "train",
                "4b",
                "roadnet_p4b_lt.json",
                "flow.json"
            ],
            "hangzhou_shenban_shixiang_1h_11_12_1530": [
                "train",
                "4b",
                "roadnet_p4b_lt.json",
                "flow.json"
            ],
            "hangzhou_tianmushan_xueyuan_1h_12_13_1664": [
                "train",
                "4b",
                "roadnet_p4b_lt.json",
                "flow.json"
            ],
            "hangzhou_baochu_tiyuchang_1h_13_14_1798": [
                "train",
                "6a",
                "roadnet_p6a_lt.json",
                "flow.json"
            ],
            "hangzhou_qingchun_yanan_1h_18_19_1544": [
                "train",
                "6a",
                "roadnet_p6a_lt.json",
                "flow.json"
            ],
            "hangzhou_qingchun_yanan_1h_17_18_1497": [
                "train",
                "6a",
                "roadnet_p6a_lt.json",
                "flow.json"
            ],
            "hangzhou_baochu_tiyuchang_1h_14_15_1865": [
                "train",
                "6a",
                "roadnet_p6a_lt.json",
                "flow.json"
            ],
            "hangzhou_shenban_shixiang_1h_1309": [
                "train",
                "6c",
                "roadnet_p6c_lt.json",
                "flow.json"
            ],
            "hangzhou_baochu_tiyuchang_1h_15_16_1931": [
                "train",
                "6c",
                "roadnet_p6c_lt.json",
                "flow.json"
            ],
            "hangzhou_tianmushan_xueyuan_1h_1247": [
                "train",
                "6c",
                "roadnet_p6c_lt.json",
                "flow.json"
            ],
            "hangzhou_tianmushan_xueyuan_1h_14_15_1868": [
                "train",
                "6c",
                "roadnet_p6c_lt.json",
                "flow.json"
            ],
            "hangzhou_qingchun_yanan_1h_7_8_1289": [
                "train",
                "6e",
                "roadnet_p6e_lt.json",
                "flow.json"
            ],
            "hangzhou_shenban_shixiang_1h_17_18_1772": [
                "train",
                "6e",
                "roadnet_p6e_lt.json",
                "flow.json"
            ],
            "hangzhou_qingchun_yanan_1h_11_12_1453": [
                "train",
                "6e",
                "roadnet_p6e_lt.json",
                "flow.json"
            ],
            "hangzhou_tianmushan_xueyuan_1h_15_16_1957": [
                "train",
                "6e",
                "roadnet_p6e_lt.json",
                "flow.json"
            ],
            "hangzhou_qingchun_yanan_1h_10_11_1406": [
                "train",
                "8",
                "roadnet_p8_lt.json",
                "flow.json"
            ],
            "hangzhou_shenban_shixiang_1h_15_16_1655": [
                "train",
                "8",
                "roadnet_p8_lt.json",
                "flow.json"
            ],
            "hangzhou_baochu_tiyuchang_1h_17_18_2108": [
                "train",
                "8",
                "roadnet_p8_lt.json",
                "flow.json"
            ],
            "hangzhou_baochu_tiyuchang_1h_18_19_1770": [
                "train",
                "8",
                "roadnet_p8_lt.json",
                "flow.json"
            ],
            "hangzhou_tianmushan_xueyuan_1h_18_19_1484": [
                "train",
                "8",
                "roadnet_p8_lt.json",
                "flow.json"
            ],
            "hangzhou_tianmushan_xueyuan_1h_8_9_2159": [
                "train",
                "8",
                "roadnet_p8_lt.json",
                "flow.json"
            ],
            "hangzhou_qingchun_yanan_1h_15_16_1554": [
                "valid",
                "4a",
                "roadnet_p4a_lt.json",
                "flow.json"
            ],
            "hangzhou_qingchun_yanan_1h_14_15_1534": [
                "valid",
                "4b",
                "roadnet_p4b_lt.json",
                "flow.json"
            ],
            "hangzhou_shenban_shixiang_1h_14_15_1614": [
                "valid",
                "6a",
                "roadnet_p6a_lt.json",
                "flow.json"
            ],
            "hangzhou_shenban_shixiang_1h_13_14_1653": [
                "valid",
                "6c",
                "roadnet_p6c_lt.json",
                "flow.json"
            ],
            "hangzhou_shenban_shixiang_1h_18_19_1489": [
                "valid",
                "6e",
                "roadnet_p6e_lt.json",
                "flow.json"
            ],
            "hangzhou_qingchun_yanan_1h_12_13_1373": [
                "valid",
                "8",
                "roadnet_p8_lt.json",
                "flow.json"
            ],
            "hangzhou_qingchun_yanan_1h_13_14_1536": [
                "test",
                "4c",
                "roadnet_p4c_lt.json",
                "flow.json"
            ],
            "hangzhou_tianmushan_xueyuan_1h_9_10_2178": [
                "test",
                "4d",
                "roadnet_p4d_lt.json",
                "flow.json"
            ],
            "hangzhou_baochu_tiyuchang_1h_8_9_2231": [
                "test",
                "6b",
                "roadnet_p6b_lt.json",
                "flow.json"
            ],
            "hangzhou_shenban_shixiang_1h_8_9_2032": [
                "test",
                "6d",
                "roadnet_p6d_lt.json",
                "flow.json"
            ],
            "hangzhou_tianmushan_xueyuan_1h_17_18_2062": [
                "test",
                "6f",
                "roadnet_p6f_lt.json",
                "flow.json"
            ],
            "real-3701022106-1h-1350": [
                "city",
                "4a",
                "roadnet_p4a_lt.json",
                "flow.json"
            ],
            "ngsim_lsr_inter_1_3600_2172": [
                "city",
                "4a",
                "roadnet_p4a_lt.json",
                "flow.json"
            ],
            "ngsim_lsr_inter_0_3600_1786": [
                "city",
                "4a",
                "roadnet_p4a_lt.json",
                "flow.json"
            ],
            "real-3701022124-1h-2255": [
                "city",
                "4b",
                "roadnet_p4b_lt.json",
                "flow.json"
            ],
            "real-3701055130-1h-1917": [
                "city",
                "6a",
                "roadnet_p6a_lt.json",
                "flow.json"
            ],
            "ngsim_lsr_inter_3_3600_2264": [
                "city",
                "6a",
                "roadnet_p6a_lt.json",
                "flow.json"
            ],
            "demo_train_1364": [
                "demo_train_1364",
                "8",
                "roadnet_1_1.json",
                "inter_4_1364.json"
            ]
        }
    },
    "DIC_REWARD_INFO": {
        "stop_vehicle_thres1": -0.25
    },
    "VALID_THRESHOLD": 30,
    "MIN_ACTION_TIME": 10,
    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "INTERVAL": 1,
    "THREADNUM": 1,
    "RLTRAFFICLIGHT": True,
    "LIST_STATE_FEATURE": [
        "cur_phase",
        "lane_num_vehicle"
    ],
    "TRAFFIC_IN_TASKS": [
        "hangzhou_baochu_tiyuchang_1h_10_11_2021",
        "hangzhou_shenban_shixiang_1h_12_13_1448",
        "hangzhou_baochu_tiyuchang_1h_12_13_1573",
        "hangzhou_tianmushan_xueyuan_1h_11_12_1744",
        "hangzhou_qingchun_yanan_1h_16_17_1504",
        "hangzhou_tianmushan_xueyuan_1h_10_11_2055",
        "hangzhou_shenban_shixiang_1h_11_12_1530",
        "hangzhou_tianmushan_xueyuan_1h_12_13_1664",
        "hangzhou_baochu_tiyuchang_1h_13_14_1798",
        "hangzhou_qingchun_yanan_1h_18_19_1544",
        "hangzhou_qingchun_yanan_1h_17_18_1497",
        "hangzhou_baochu_tiyuchang_1h_14_15_1865",
        "hangzhou_shenban_shixiang_1h_1309",
        "hangzhou_baochu_tiyuchang_1h_15_16_1931",
        "hangzhou_tianmushan_xueyuan_1h_1247",
        "hangzhou_tianmushan_xueyuan_1h_14_15_1868",
        "hangzhou_qingchun_yanan_1h_7_8_1289",
        "hangzhou_shenban_shixiang_1h_17_18_1772",
        "hangzhou_qingchun_yanan_1h_11_12_1453",
        "hangzhou_tianmushan_xueyuan_1h_15_16_1957",
        "hangzhou_qingchun_yanan_1h_10_11_1406",
        "hangzhou_shenban_shixiang_1h_15_16_1655",
        "hangzhou_baochu_tiyuchang_1h_17_18_2108",
        "hangzhou_baochu_tiyuchang_1h_18_19_1770",
        "hangzhou_tianmushan_xueyuan_1h_18_19_1484",
        "hangzhou_tianmushan_xueyuan_1h_8_9_2159"
    ],
    "ROADNET_FILE": "roadnet_1_1.json",
    "FLOW_FILE": "inter_4_1364.json",
    "LANE_PHASE_INFOS": {
        "intersection_1_1": {
            "start_lane": ["road_0_1_0_0", "road_0_1_0_1",
                           "road_1_0_1_0", "road_1_0_1_1",
                           "road_1_2_3_0", "road_1_2_3_1",
                           "road_2_1_2_0", "road_2_1_2_1"],
            "same_start_lane": [["road_0_1_0_0"], ["road_0_1_0_1"],
                                ["road_1_0_1_0"], ["road_1_0_1_1"],
                                ["road_1_2_3_0"], ["road_1_2_3_1"],
                                ["road_2_1_2_0"], ["road_2_1_2_1"]],
            "end_lane": ["road_1_1_0_0", "road_1_1_0_1",
                         "road_1_1_1_0", "road_1_1_1_1",
                         "road_1_1_2_0", "road_1_1_2_1",
                         "road_1_1_3_0", "road_1_1_3_1"],
            "phase": [1, 2, 3, 4, 5, 6, 7, 8],
            "yellow_phase": 0,
            "phase_startLane_mapping": {1: ["road_0_1_0_1", "road_2_1_2_1"],
                                        2: ["road_1_0_1_1", "road_1_2_3_1"],
                                        3: ["road_0_1_0_0", "road_2_1_2_0"],
                                        4: ["road_1_0_1_0", "road_1_2_3_0"],
                                        5: ["road_0_1_0_1", "road_0_1_0_0"],
                                        6: ["road_2_1_2_1", "road_2_1_2_0"],
                                        7: ["road_1_0_1_1", "road_1_0_1_0"],
                                        8: ["road_1_2_3_0", "road_1_2_3_1"]
                                        },
            "phase_noRightStartLane_mapping": {
                "1": ["road_0_1_0_1", "road_2_1_2_1"],
                "2": ["road_1_0_1_1", "road_1_2_3_1"],
                "3": ["road_0_1_0_0", "road_2_1_2_0"],
                "4": ["road_1_0_1_0", "road_1_2_3_0"],
                "5": ["road_0_1_0_1", "road_0_1_0_0"],
                "6": ["road_2_1_2_1", "road_2_1_2_0"],
                "7": ["road_1_0_1_1", "road_1_0_1_0"],
                "8": ["road_1_2_3_0", "road_1_2_3_1"]
            },
            "phase_sameStartLane_mapping": {
                "1": [["road_0_1_0_1"], ["road_2_1_2_1"]],
                "2": [["road_1_0_1_1"], ["road_1_2_3_1"]],
                "3": [["road_0_1_0_0"], ["road_2_1_2_0"]],
                "4": [["road_1_0_1_0"], ["road_1_2_3_0"]],
                "5": [["road_0_1_0_1"], ["road_0_1_0_0"]],
                "6": [["road_2_1_2_1"], ["road_2_1_2_0"]],
                "7": [["road_1_0_1_1"], ["road_1_0_1_0"]],
                "8": [["road_1_2_3_0"], ["road_1_2_3_1"]]
            },
            "phase_roadLink_mapping": {
                "1": [["road_0_1_0_1", "road_1_1_0_1", "go_straight"],
                      ["road_0_1_0_1", "road_1_1_0_0", "go_straight"],
                      ["road_2_1_2_1", "road_1_1_2_1", "go_straight"],
                      ["road_2_1_2_1", "road_1_1_2_0", "go_straight"]],
                "2": [["road_1_2_3_1", "road_1_1_3_0", "go_straight"],
                      ["road_1_0_1_1", "road_1_1_1_1", "go_straight"],
                      ["road_1_2_3_1", "road_1_1_3_1", "go_straight"],
                      ["road_1_0_1_1", "road_1_1_1_0", "go_straight"]],
                "3": [["road_0_1_0_0", "road_1_1_1_0", "turn_left"],
                      ["road_2_1_2_0", "road_1_1_3_0", "turn_left"],
                      ["road_0_1_0_0", "road_1_1_1_1", "turn_left"],
                      ["road_2_1_2_0", "road_1_1_3_1", "turn_left"]],
                "4": [["road_1_0_1_0", "road_1_1_2_1", "turn_left"],
                      ["road_1_0_1_0", "road_1_1_2_0", "turn_left"],
                      ["road_1_2_3_0", "road_1_1_0_1", "turn_left"],
                      ["road_1_2_3_0", "road_1_1_0_0", "turn_left"]],
                "5": [["road_0_1_0_1", "road_1_1_0_1", "go_straight"],
                      ["road_0_1_0_1", "road_1_1_0_0", "go_straight"],
                      ["road_0_1_0_0", "road_1_1_1_1", "turn_left"],
                      ["road_0_1_0_0", "road_1_1_1_0", "turn_left"]],
                "6": [["road_2_1_2_1", "road_1_1_2_1", "go_straight"],
                      ["road_2_1_2_0", "road_1_1_3_0", "turn_left"],
                      ["road_2_1_2_0", "road_1_1_3_1", "turn_left"],
                      ["road_2_1_2_1", "road_1_1_2_0", "go_straight"]],
                "7": [["road_1_0_1_0", "road_1_1_2_1", "turn_left"],
                      ["road_1_0_1_0", "road_1_1_2_0", "turn_left"],
                      ["road_1_0_1_1", "road_1_1_1_1", "go_straight"],
                      ["road_1_0_1_1", "road_1_1_1_0", "go_straight"]],
                "8": [["road_1_2_3_1", "road_1_1_3_0", "go_straight"],
                      ["road_1_2_3_1", "road_1_1_3_1", "go_straight"],
                      ["road_1_2_3_0", "road_1_1_0_1", "turn_left"],
                      ["road_1_2_3_0", "road_1_1_0_0", "turn_left"]]
            },
            "relation": [[0, 0, 0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 1],
                         [0, 0, 0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 1],
                         [1, 0, 1, 0, 0, 0, 0],
                         [1, 0, 1, 0, 0, 0, 0],
                         [0, 1, 0, 1, 0, 0, 0],
                         [0, 1, 0, 1, 0, 0, 0]]
        }
    },
    "INTER_NAME": None,
    "LANE_PHASE_INFO": None
}
