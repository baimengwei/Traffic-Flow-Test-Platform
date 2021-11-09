from copy import deepcopy


class ConfAgent(dict):
    def __init__(self, args):
        config_name = "DIC_AGENT_CONF_" + args.algorithm.upper()
        self.__config = getattr(Constrant, config_name)
        super().__init__(self.__config)
        self.__delattr__('_ConfAgent__config')
        pass

    def __repr__(self):
        return "ConfAgent"


class Constrant:
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
        "TIME_PHASE": 15,
    }

    DIC_AGENT_CONF_MAXPRESSURE = {
        "G_MIN": 5,
    }

    DIC_AGENT_CONF_DQN = {
        "SAMPLE_SIZE": 5000,
        "BATCH_SIZE": 32,
        "EPOCHS": 100,
        "UPDATE_Q_BAR_FREQ": 5,
        "GAMMA": 0.8,
        "MAX_MEMORY_LEN": 50000,
        "EPSILON": 0.8,
        "EPSILON_DECAY": 0.90,
        "MIN_EPSILON": 0.2,
    }

    DIC_AGENT_CONF_DRQN = deepcopy(DIC_AGENT_CONF_DQN)
    DIC_AGENT_CONF_DRQN.update({
        "HISTORY_LEN": 20,
        "HIDDEN_DIM": 10
    })

    DIC_AGENT_CONF_FRAP = deepcopy(DIC_AGENT_CONF_DQN)

    DIC_AGENT_CONF_FRAPRQ = deepcopy(DIC_AGENT_CONF_DRQN)

    DIC_AGENT_CONF_METADQN = {
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

    DIC_AGENT_CONF_TDDD = {
        "LR": 0.001,
        "LR_ACTOR": 0.0001,
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


if __name__ == '__main__':
    class Args:
        algorithm = 'DQN'


    args = Args()
    x = ConfAgent(args)
    print(x)
