#!/usr/bin/env python
# encoding: utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: data_structures.py
@time: 2020/4/23 11:12
@desc:
'''


class Option(object):
    def __init__(self):
        pass


class ConvertBond(object):
    def __init__(self, conv_bond_sec_id, equ_sec_id, total_size, strike_price):
        self.conv_bond = conv_bond_sec_id
        self.equ_sec_id = equ_sec_id
        self.total_size = total_size
        self.strike_price = strike_price

    def initialize(self, trade_date, start_date, end_date):
        pass

    def cache_conv_bond_mkts(self):
        pass

    def cache_equ_mkts(self):
        pass

    def cv_pricer(self):
        pass
