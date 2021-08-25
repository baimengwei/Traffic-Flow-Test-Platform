from configs.traffic_meta_train import *
from configs.traffic_meta_test import *

TRAFFIC_CATEGORY = {
    "train": {  # meta-train
        "4a": paper_meta_train_4a_phase_traffic_list,
        "4b": paper_meta_train_4b_phase_traffic_list,
        "6a": paper_meta_train_6a_phase_traffic_list,
        "6c": paper_meta_train_6c_phase_traffic_list,
        "6e": paper_meta_train_6e_phase_traffic_list,
        "8": paper_meta_train_8_phase_traffic_list
    },
    "valid": {  # meta-test task1
        "4a": paper_meta_valid_4a_phase_traffic_list,
        "4b": paper_meta_valid_4b_phase_traffic_list,
        "6a": paper_meta_valid_6a_phase_traffic_list,
        "6c": paper_meta_valid_6c_phase_traffic_list,
        "6e": paper_meta_valid_6e_phase_traffic_list,
        "8": paper_meta_valid_8_phase_traffic_list
    },
    "test": {  # meta-test task2
        "4c": paper_meta_test_4c_traffic_list,
        "4d": paper_meta_test_4d_traffic_list,
        "6b": paper_meta_test_6b_traffic_list,
        "6d": paper_meta_test_6d_traffic_list,
        "6f": paper_meta_test_6f_traffic_list
    },
    "city": city_train_phase,
    # manually change (meta-test task3 homo) # 6 files
    # "city": city_test_phase, # manually change (meta-test task3 here) # 10
    # files
}

roadnet_map = {
    "4a": "roadnet_p4a_lt.json",
    "4b": "roadnet_p4b_lt.json",
    "4c": "roadnet_p4c_lt.json",
    "4d": "roadnet_p4d_lt.json",

    "6a": "roadnet_p6a_lt.json",
    "6b": "roadnet_p6b_lt.json",
    "6c": "roadnet_p6c_lt.json",
    "6d": "roadnet_p6d_lt.json",
    "6e": "roadnet_p6e_lt.json",
    "6f": "roadnet_p6f_lt.json",

    "8": "roadnet_p8_lt.json",
}

flow_map = {
    "4": "flow.json",
    "3e": "flow_3e.json",
    "3n": "flow_3n.json",
    "3s": "flow_3s.json",
    "3w": "flow_3w.json"
}

meta_train_traffic = [t for type in TRAFFIC_CATEGORY["train"]
                      for t in TRAFFIC_CATEGORY["train"][type]]
meta_valid_traffic = [t for type in TRAFFIC_CATEGORY["valid"]
                      for t in TRAFFIC_CATEGORY["valid"][type]]
meta_test_traffic = [t for type in TRAFFIC_CATEGORY["test"]
                     for t in TRAFFIC_CATEGORY["test"][type]]
meta_test_city = [t for type in TRAFFIC_CATEGORY["city"]
                  for t in TRAFFIC_CATEGORY["city"][type]]

TRAFFIC_CATEGORY["train_all"] = meta_train_traffic
TRAFFIC_CATEGORY["valid_all"] = meta_valid_traffic
TRAFFIC_CATEGORY["test_all"] = meta_test_traffic
TRAFFIC_CATEGORY["city_all"] = meta_test_city

TRAFFIC_CATEGORY["traffic_info"] = {}
# make all files don't be duplicated, and get a log_dict in traffic_category
for ctg in ["train", "valid", "test", "city"]:
    for type in TRAFFIC_CATEGORY[ctg].keys():
        if "3" in roadnet_map[type]:
            flow_file = flow_map[roadnet_map[type][5:7]]
        else:
            flow_file = flow_map["4"]
        for traffic in TRAFFIC_CATEGORY[ctg][type]:
            if traffic in TRAFFIC_CATEGORY["traffic_info"].keys():
                print("old: ", TRAFFIC_CATEGORY["traffic_info"][traffic])
                print("new: ", (ctg, type, roadnet_map[type], flow_file))
                print("traffic info is already set! Attention duplicate!")
                raise ValueError
            TRAFFIC_CATEGORY["traffic_info"][traffic] = \
                (ctg, type, roadnet_map[type], flow_file)

TRAFFIC_CATEGORY["traffic_info"]["demo_train_1364"] = ('demo_train_1364', '8',
                                                  'roadnet_1_1.json',
                                                  'inter_4_1364.json')
