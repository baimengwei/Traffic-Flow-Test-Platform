from configs.config_phaser import parse


def run_metalight():
    from algs.MetaLight.metalight_train import main
    args = parse()
    print('start execute...')
    args.algorithm = 'MetaLight'
    args.memo = 'MetaLight_XXGX'
    main(args)


def run_frapplus():
    from algs.FRAPPlus.frapplus_train import main
    args = parse()
    print('start execute frapplus...')
    args.algorithm = 'FRAPPlus'
    args.memo = 'FRAPPlus_Min'
    main(args)


def run_frap():
    from algs.FRAP.frap_train import main
    args = parse()
    print('start execute frap...')
    args.algorithm = 'FRAP'
    args.memo = 'FRAP_MMM'
    main(args)


# TODO here
# def run_fraprq():
#     from algs.FRAPRQ.fraprq_train import main
#     args = parse()
#     print('start execute fraprq...')
#     args.algorithm = 'FRAPRQ'
#     args.memo = 'FRAP_RQ_T'
#     main(args)

def run_dqn():
    from algs.DQN.dqn_train import main
    args = parse()
    print('start execute dqn...')
    args.algorithm = "DQN"
    args.memo = "DQN_C"
    main(args)


def run_sotl():
    from algs.SOTL.sotl_train import main
    args = parse()
    print('start execute sotl...')
    args.algorithm = "SOTL"
    args.memo = "SOTL_AA"
    main(args)


def run_tddd():
    from algs.TDDD.tddd_train import main
    args = parse()
    print('start execute tddd')
    args.algorithm = "TDDD"
    args.memo = "TDDD_ASD"
    main(args)


def run_drqn():
    from algs.DRQN.drqn_train import main
    args = parse()
    print('start execute drqn')
    args.algorithm = "DRQN"
    args.memo = "DRQN_PJ"
    main(args)


def run_webster():
    from algs.WEBSTER.webster_train import main
    args = parse()
    print('start execute webster..')
    args.algorithm = "WEBSTER"
    args.memo = "WEBSTER_OZ"
    main(args)


def run_fixtime():
    from algs.FIXTIME.fixtime_train import main
    args = parse()
    print('start execute fixtime..')
    args.algorithm = "FIXTIME"
    args.memo = "FIXTIME_AS"
    main(args)

def run_maxpressure():
    from algs.MAXPRESSURE.maxpressure_train import main
    args = parse()
    print("start execute maxpressure.")
    args.algorithm = "MAXPRESSURE"
    args.memo = "MAXPRESSURE_N"
    main(args)

if __name__ == '__main__':
    # run_metalight()
    # run_frap()
    # run_frapplus()
    # run_dqn()
    # run_sotl()
    # run_tddd()
    # run_drqn()
    # run_webster()
    # run_fixtime()
    run_maxpressure()
