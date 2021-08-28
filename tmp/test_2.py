import multiprocessing.pool
from contextlib import closing
from functools import partial

class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class Pool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

def foo(x, depth=0):
    if depth == 0:
        return x
    else:
        with closing(Pool()) as p:
            return p.map(partial(foo, depth=depth-1), range(x + 1))

if __name__ == "__main__":
    from pprint import pprint
    pprint(foo(10, depth=2))