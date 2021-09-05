from multiprocessing import *
import time


def func_xx(i, x):
    print('this is a func_xx ', i, x)
    time.sleep(3)


def func(i):
    print('this is a func ', i)
    list_p = []
    for x in range(3):
        p = Process(target=func_xx, args=(i, x,))
        list_p.append(p)
        p.start()
    for p in list_p:
        p.join()
    time.sleep(1)


def aaai():
    list_p = []
    for i in range(5):
        p = Process(target=func, args=(i,))
        p.start()
        list_p.append(p)

    for p in list_p:
        p.join()
    print("finished.")


if __name__ == '__main__':
    aaai()
