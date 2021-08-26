from configs.config_phaser import parse


def run_metalight():
    from algs.MetaLight.meta_train import main
    args = parse()
    print('start execute...')
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


if __name__ == '__main__':
    # run_metalight()
    # run_frap()
    run_frapplus()
