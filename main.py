from configs.config_phaser import parse


def run_metalight():
    from algs.MetaLight.metalight_train import main
    args = parse()
    print('start execute...')
    args.algorithm = 'MetaLight'
    args.project = 'MetaLight_XXGX'
    main(args)


def run_frapplus():
    from algs.FRAPPLUS.frapplus_train import main
    args = parse()
    print('start execute frapplus...')
    args.algorithm = 'FRAPPlus'
    args.project = 'FRAPPlus_Min'
    main(args)


def run_dqn():
    from algs.DQN.dqn_train import main
    args = parse()
    print('start execute dqn...')
    args.algorithm = "DQN"
    args.project = "DQN_TEST"
    args.env = "cityflow"
    main(args)


def run_metadqn():
    # from algs.MetaDQN.metadqn_train import main
    from algs.MetaDQN.metadqn_train import main_test
    args = parse()
    print('start execute meta dqn...')
    args.algorithm = "MetaDQN"
    args.project = "Meta_DQN_RD"
    args.env = "anno"
    # main(args)
    args.train_round = args.adapt_round
    args.algorithm = "MetaDQNAdapt"
    main_test(args)


def run_sotl():
    from algs.SOTL.sotl_train import main
    args = parse()
    print('start execute sotl...')
    args.algorithm = "SOTL"
    args.project = "SOTL_AA"
    main(args)


def run_tddd():
    from algs.TDDD.tddd_train import main
    args = parse()
    print('start execute tddd')
    args.algorithm = "TDDD"
    args.project = "TDDD_ASD"
    main(args)


def run_drqn():
    from algs.DRQN.drqn_train import main
    args = parse()
    print('start execute drqn')
    args.algorithm = "DRQN"
    args.project = "DRQN_PJ"
    main(args)


def run_fraprq():
    from algs.FRAPRQ.fraprq_train import main
    args = parse()
    print('start execute fraprq')
    args.algorithm = "FRAPRQ"
    args.project = "FRAPRQ_MSP"
    main(args)


def run_webster():
    from algs.WEBSTER.webster_train import main
    args = parse()
    print('start execute webster..')
    args.algorithm = "WEBSTER"
    args.project = "WEBSTER_OZ"
    main(args)


def run_fixtime():
    from algs.FIXTIME.fixtime_train import main
    args = parse()
    print('start execute fixtime..')
    args.algorithm = "FIXTIME"
    args.project = "FIXTIME_AS"
    args.train_round = 1
    main(args)


def run_maxpressure():
    from algs.MAXPRESSURE.maxpressure_train import main
    args = parse()
    print("start execute maxpressure.")
    args.algorithm = "MAXPRESSURE"
    args.project = "MAXPRESSURE_N"
    args.env = "cityflow"
    main(args)

def run_ql():
    from algs.QL.ql_train import main
    args = parse()
    print("start execute q-learning.")
    args.algorithm = "QL"
    args.project = "QL_TTT"
    args.env = "cityflow"
    main(args)

def run_gsqldsep():
    from algs.GSQLDSEP.gsqldsep_train import main
    args = parse()
    print("start execute gsqldsep-learning.")
    args.algorithm = "GSQLDSEP"
    args.project = "GSQLDSEP_TAX"
    args.env = "cityflow"
    main(args)

def run_sql():
    from algs.SQL.sql_train import main
    args = parse()
    print("start execute sql.")
    args.algorithm = "SQL"
    args.project = "SQL_EMM"
    args.env = "cityflow"
    main(args)

def run_gsql():
    from algs.GSQL.gsql_train import main
    args = parse()
    print("start execute gsql.")
    args.algorithm = "GSQL"
    args.project = "GSQL_EMM"
    args.env = "cityflow"
    main(args)

def run_dynaq():
    from algs.DYNAQ.dynaq_train import main
    args = parse()
    print("start execute gsql.")
    args.algorithm = "DYNAQ"
    args.project = "DYNAQ_EMM"
    args.env = "cityflow"
    main(args)

if __name__ == '__main__':
    # from pycallgraph import PyCallGraph
    # from pycallgraph.output import GraphvizOutput
    # from pycallgraph import Config
    # from pycallgraph import GlobbingFilter
    #
    # config = Config()
    # config.trace_filter = GlobbingFilter(include=['*'])
    # graphviz = GraphvizOutput()
    # graphviz.output_file = 'graph.png'
    # with PyCallGraph(output=graphviz, config=config):
    #     run_gsql()
    # run_ql()
    # run_gsqldsep()
    # run_sql()
    run_gsql()
    # run_dynaq()
    # run_metalight()
    # run_frapplus()
    # run_dqn()
    # run_sotl()
    # run_tddd()
    # run_drqn()
    # run_webster()
    # run_fixtime()
    # run_maxpressure()
    # run_fraprq()
    # run_dqn()
    # run_metadqn()
