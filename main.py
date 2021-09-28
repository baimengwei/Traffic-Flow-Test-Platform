from configs.config_phaser import parse


def run_metalight():
    from algs.MetaLight.metalight_train import main
    args = parse()
    print('start execute...')
    args.algorithm = 'MetaLight'
    args.project = 'MetaLight_XXGX'
    main(args)


def run_frapplus():
    from algs.FRAPPlus.frapplus_train import main
    args = parse()
    print('start execute frapplus...')
    args.algorithm = 'FRAPPlus'
    args.project = 'FRAPPlus_Min'
    main(args)


def run_frap():
    from algs.FRAP.frap_train import main
    args = parse()
    print('start execute frap...')
    args.algorithm = 'FRAP'
    args.project = 'FRAP_MMM'
    main(args)


def run_dqn():
    from algs.DQN.dqn_train import main
    args = parse()
    print('start execute dqn...')
    args.algorithm = "DQN"
    args.project = "DQN_Meta2"
    args.env = "anno"
    main(args)


def run_metadqn():
    from algs.MetaDQN.metadqn_train import main
    args = parse()
    print('start execute meta dqn...')
    args.algorithm = "MetaDQN"
    args.project = "Meta_DQN_RD"
    args.env = "anno"
    main(args)

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
    main(args)


def run_maxpressure():
    from algs.MAXPRESSURE.maxpressure_train import main
    args = parse()
    print("start execute maxpressure.")
    args.algorithm = "MAXPRESSURE"
    args.project = "MAXPRESSURE_N"
    args.env = "sumo"
    args.if_gui = False
    main(args)


if __name__ == '__main__':
    # run_metalight()
    # run_frap()
    # run_frapplus()
    run_dqn()
    # run_sotl()
    # run_tddd()
    # run_drqn()
    # run_webster()
    # run_fixtime()
    # run_maxpressure()
    # run_fraprq()
    # run_metadqn()
