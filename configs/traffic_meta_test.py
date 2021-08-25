"""
    meta-test taks-1 homo traffic file: {4a, 4b, 6a, 6a, 6c, 6e}
"""

paper_meta_valid_4a_phase_traffic_list = [
    "hangzhou_qingchun_yanan_1h_15_16_1554",
]

paper_meta_valid_4b_phase_traffic_list = [
    "hangzhou_qingchun_yanan_1h_14_15_1534",
]

paper_meta_valid_6a_phase_traffic_list = [
    "hangzhou_shenban_shixiang_1h_14_15_1614",
]

paper_meta_valid_6c_phase_traffic_list = [
    "hangzhou_shenban_shixiang_1h_13_14_1653",
]

paper_meta_valid_6e_phase_traffic_list = [
    "hangzhou_shenban_shixiang_1h_18_19_1489",
]

paper_meta_valid_8_phase_traffic_list = [
    "hangzhou_qingchun_yanan_1h_12_13_1373",
]


"""
    meta-test task2: Heme. traffic file: {4c, 4d, 6b, 6d, 6f}
"""

paper_meta_test_4c_traffic_list = [
    "hangzhou_qingchun_yanan_1h_13_14_1536",
]

paper_meta_test_4d_traffic_list = [
    "hangzhou_tianmushan_xueyuan_1h_9_10_2178"
]

paper_meta_test_6b_traffic_list = [
    "hangzhou_baochu_tiyuchang_1h_8_9_2231",
]

paper_meta_test_6d_traffic_list = [
    "hangzhou_shenban_shixiang_1h_8_9_2032"
]

paper_meta_test_6f_traffic_list = [
    "hangzhou_tianmushan_xueyuan_1h_17_18_2062",
]
# valid 6 files, test 5 files
"""
    Name of traffic fiel 
    Jinan: real** 
    Atlanta : ngsim*(1364, 2172, 2244, 2264)
    LA      : ngsim*(1934, 1786, 1790)
    meta-test task3 different cities Homo. traffic file {4a, 4b, 6a}
"""

paper_meta_test_4a_different_cities = [
    "real-3701022106-1h-1350",
    "ngsim_lsr_inter_1_3600_2172",
    "ngsim_lsr_inter_0_3600_1786",
]

paper_meta_test_4b_different_cities = [
    "real-3701022124-1h-2255",
]

paper_meta_test_6a_different_cities = [
    "real-3701055130-1h-1917",
    "ngsim_lsr_inter_3_3600_2264",
]

""""
    meta-test task3 different cities Heme. traffic file {4c, 4d, 6d, 6f}
"""

paper_meta_test_4c_different_cities = [
    "real-3701022111-1h-1691",
    "ngsim_lsr_inter_1_3600_2172",
    "ngsim_lsr_inter_0_3600_1786",
]

paper_meta_test_4d_different_cities = [
    "real-3701126122-1h-1990",
    "ngsim_lsr_inter_4_3600_1364",
]

paper_meta_test_6d_different_cities = [
    "real-3701033113-1h-2078",
    "ngsim_lsr_inter_2_3600_1790",
]

paper_meta_test_6f_different_cities = [
    "real-3701033122-1h-1722",
    "ngsim_lsr_inter_2_3600_2244",
    "ngsim_lsr_inter_3_3600_1934",
]

city_train_phase = {
        "4a": paper_meta_test_4a_different_cities,
        "4b": paper_meta_test_4b_different_cities,
        "6a": paper_meta_test_6a_different_cities,
    }

city_test_phase = {
        "4c": paper_meta_test_4c_different_cities,
        "4d": paper_meta_test_4d_different_cities,
        "6d": paper_meta_test_6d_different_cities,
        "6f": paper_meta_test_6f_different_cities
    }