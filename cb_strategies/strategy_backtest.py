#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/17 16:22
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : strategy_backtest.py

import os
import pprint
import pandas as pd


def get_ids(df: pd.DataFrame, equ_weight=0.7, topN=10, last_codes: list = []):
    # ['代码', '名称', '最新', '涨跌幅', '成交量(手)', '成交金额', '换手率', '正股名称', '正股价格',
    #  '正股涨跌幅', '转股价格', '转股价值', '转股溢价率', '纯债价值', '纯债YTM', '平价底价溢价率', '权证价格',
    #  '有条件赎回价', '到期赎回价', '有条件回售价', '剩余期限', '到期日期', '债券余额(亿元)', '主体评级', '债券评级',
    #  '类型', '双低', 'pure_debt_prem', 'debt_rank', 'equity_rank', 'rank']

    # ['sec_code', 'sec_name', 'last_price', 'chg', 'vol', 'value', 'turn', 'equ_name', 'equ_price',
    #  'equ_chg', 'strike_price', 'equ_value', 'equ_premium', 'debt_premium', 'YTM', 'price_premuim_rate', 'option_price',
    #  'cond_repurchase_price', 'deal_repurchase_price', 'cond_sale_price', 'remain_dates', 'deal_dates', 'debt_value',
    #  'rating', 'debt_rating',
    #  'type', 'double_rank', 'pure_debt_prem', 'debt_rank', 'equity_rank', 'rank']
    df = df.iloc[:, :31]
    df.columns = ['sec_code', 'sec_name', 'last_price', 'chg', 'vol', 'value', 'turn', 'equ_name', 'equ_price',
                  'equ_chg', 'strike_price', 'equ_value', 'equ_premium', 'debt_premium', 'YTM', 'price_premuim_rate',
                  'option_price',
                  'cond_repurchase_price', 'deal_repurchase_price', 'cond_sale_price', 'remain_dates', 'deal_dates',
                  'debt_value', 'rating', 'debt_rating',
                  'type', 'double_rank', 'pure_debt_prem', 'debt_rank', 'equity_rank', 'rank']
    print(df.columns)
    # print(df.columns)
    _turn_median = df.iloc[:, 6].astype(float).median()
    last_price = []
    for item in df.values:
        if item[0] in last_codes:
            last_price.append(item[2])
    if len(last_price) != len(last_codes):
        print("check, not same len")
    df['score'] = equ_weight * df['equity_rank'].astype(float) + (1 - equ_weight) * df['debt_rank'].astype(float)

    df = df.loc[[(not str(item).startswith('132')) for item in df['sec_code']]]
    df = df.loc[df['cond_sale_price'] == 'inf']
    df = df.loc[df['cond_repurchase_price'] == 'inf']
    df = df.loc[[item > 1.0 for item in df['remain_dates'].astype(float)]]
    df = df.loc[[item > 0 for item in df['last_price'].astype(float)]]
    df = df.loc[[item < 115 for item in df['last_price'].astype(float)]]
    df = df.loc[[float(item) > _turn_median for item in df['turn']]]

    df = df.sort_values(by='score', ascending=True)
    df_code = df.iloc[:topN, 0]
    df_price = df.iloc[:topN, 2]
    return list(df_code), list(df_price), list(last_price)


if __name__ == "__main__":
    l = os.listdir("C:/projects/pycharm/converted_bond_strategies/cb_strategies/cache")
    l1 = ["C:/projects/pycharm/converted_bond_strategies/cb_strategies/cache/" + item for item in l if 'rank' in item]
    # pprint.pprint(l1)
    last_codes = []
    ret = []
    for f in l1:
        # print(f)
        ret_code, ret_price, last_price = get_ids(pd.read_csv(f, dtype=object), last_codes=last_codes)
        last_codes = ret_code
        ret.append((ret_code, ret_price, last_price))
    n_len = len(ret)
    returns = []
    mv = []
    total_return = 0.0
    for idx in range(1, n_len):
        if len(ret[idx][2]) != len(ret[idx - 1][1]):  # FIXME
            continue
        _curr = sum([float(item) for item in ret[idx][2]])
        _last = sum([float(item) for item in ret[idx - 1][1]])
        print("sell price:{0}, buy price:{1}".format(_curr, _last))
        total_return += (_curr - _last)
    pprint.pprint(total_return)

    # print(ret_code)
    # print(ret_price)
    print(len(l1))
