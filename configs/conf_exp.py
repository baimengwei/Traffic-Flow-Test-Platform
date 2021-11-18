class ConfExp:
    def __init__(self, args):
        self.__model_name = args.algorithm
        self.__train_round = args.train_round
        self.__num_generators = args.num_generator
        self.__num_pipeline = args.num_pipeline

        if "META" in args.algorithm.upper():
            self.__task_round = args.task_round
            self.__task_count = args.task_count
            self.__adapt_round = args.adapt_round

    @property
    def NUM_GENERATORS(self):
        return self.__num_generators

    @property
    def TRAIN_ROUND(self):
        return self.__train_round

    @property
    def MODEL_NAME(self):
        return self.__model_name