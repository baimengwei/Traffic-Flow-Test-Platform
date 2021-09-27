from misc.utils import *
from envs.anon_env import AnonEnv
import platform
import time

if platform == "Linux":
    import cityflow


class CityFlowEnv(AnonEnv):
    def __init__(self, dic_path, dic_traffic_env_conf):
        super().__init__(dic_path, dic_traffic_env_conf)

    def reset(self):
        if not os.path.isfile(self.dic_path["PATH_TO_ROADNET_FILE"]):
            raise FileExistsError("file not exist! check it %s" %
                                  self.dic_path["PATH_TO_ROADNET_FILE"])
        if not os.path.isfile(self.dic_path["PATH_TO_FLOW_FILE"]):
            raise FileExistsError("file not exist! check it %s" %
                                  self.dic_path["PATH_TO_FLOW_FILE"])
        config_file = self._save_config_file()
        self.eng = cityflow.Engine(
            config_file, self.dic_traffic_env_conf["THREADNUM"])
        os.remove(config_file)
        self.reset_prepare()
        state = self.get_state()
        return state

    def _save_config_file(self):
        config_dict = {
            "interval": self.dic_traffic_env_conf["INTERVAL"],
            "seed": 0,
            "dir": "",
            "roadnetFile": self.dic_path["PATH_TO_ROADNET_FILE"],
            "flowFile": self.dic_path["PATH_TO_FLOW_FILE"],
            "rlTrafficLight": self.dic_traffic_env_conf["RLTRAFFICLIGHT"],
            "saveReplay": self.dic_traffic_env_conf["SAVEREPLAY"],
            "roadnetLogFile": os.path.join(self.path_to_log, "roadnet_.json"),
            "replayLogFile": os.path.join(self.path_to_log, "replay_.txt"),
        }
        config_name = str(time.time()) + str(np.random.random()) + ".tmp"
        with open(config_name, "w") as f:
            json.dump(config_dict, f)
        return config_name

    def save_replay(self):
        for inter_name in sorted(self.lane_phase_infos.keys()):
            path_to_log_file = os.path.join(
                self.path_to_log, "%s.pkl" % inter_name)
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_name], f)
            f.close()


if __name__ == '__main__':
    os.chdir('../')
    # Out of date. refresh please.
    print('env test start...')
    print('test finished..')
