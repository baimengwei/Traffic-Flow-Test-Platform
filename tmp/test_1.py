from multiprocessing import Process
import time
import multiprocessing.pool


class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class Pool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def print_run(msg):
    print('msg.%s' % msg)
    time.sleep(2)


def print_log(msg):
    p = Process(target=print_run,
                args=(msg,))
    p.start()
    p.join()
    print('msg is : %s' % msg)
    time.sleep(1)


def set_multprocess():
    pool = Pool(processes=10)
    for i in range(20):
        pool.apply_async(func=print_log, args=(str(i),))

    pool.close()
    pool.join()
    print('finished')


if __name__ == '__main__':
    set_multprocess()
