#!/usr/bin/env python
# encoding: utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: convert_bond_pricing.py
@time: 2020/4/23 11:14
@desc:
'''

import matplotlib.pyplot as plt


def handle_terms():
    pass


if __name__ == '__main__':
    y = range(10)
    xtickers = [0, 2, 4, 6, 8]
    xlabels = ['a', 'b', 'c', 'd', 'e']
    plt.xticks(xtickers, xlabels)
    plt.plot(y)
    plt.show()
