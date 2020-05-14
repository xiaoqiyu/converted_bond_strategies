#!/usr/bin/env python
# encoding: utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: helper.py
@time: 2020/5/12 22:52
@desc:
'''

import types
import time
from functools import wraps


def func_count(func):
    num = 0

    @wraps(func)
    def call_func(*args, **kwargs):
        ret = func(*args, **kwargs)
        nonlocal num
        num += 1
        print('call {0} for {1}th time with args:{2} and kwargs:{3}'.format(func.__name__, num, args, kwargs))
        return ret, num

    return call_func


def timeit(func):
    @wraps(func)
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        # logger.info('%r (%r, %r) %2.2f sec' % (func.__name__, args, kwargs, te - ts))
        print('%r (%r, %r) %2.2f sec' % (func.__name__, args, kwargs, te - ts))
        return result

    return timed


@func_count
@timeit
def add_func(x=1, y=2):
    return x + y
    # time.sleep(1)


@timeit
def test_time():
    time.sleep(2)


if __name__ == "__main__":
    ret = add_func(x=1, y=2)
    print(ret)
    ret = add_func(x=1, y=3)
    print(ret)
    ret = add_func(x=1, y=4)
    print(ret)
    # test_time()
