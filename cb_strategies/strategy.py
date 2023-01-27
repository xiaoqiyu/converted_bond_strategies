#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/27 21:10
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : strategy.py

import pandas as pd
import numpy as np

# df = pd.read_excel('cache/mkt20210627.xlsx', encoding='gbk')
# print(df.shape)

# with open('cache/mkt20210627.csv', 'r') as f:
#     ret = f.readlines()
#     print(ret[0])
#     print(len(ret))

# , encoding='utf-8'
trade_date = '20230108'
equ_ratio = 0.7
df = pd.read_csv('cache/mkt{0}.csv'.format(trade_date), encoding='gbk')

df['pure_debt_prem'] = df['最新'] / df['纯债价值'] - 1
df['debt_rank'] = df['pure_debt_prem'].rank()
df['equity_rank'] = df['转股溢价率'].rank()
df['rank'] = df['debt_rank'] * (1 - equ_ratio) + df['equity_rank'] * equ_ratio

df.columns = ['sec_code', 'sec_name', 'last_price', 'chg', 'vol', 'value', 'turn', 'equ_name', 'equ_price',
              'equ_chg', 'strike_price', 'equ_value', 'equ_premium', 'debt_premium', 'YTM', 'price_premuim_rate',
              'option_price',
              'cond_repurchase_price', 'deal_repurchase_price', 'cond_sale_price', 'remain_dates', 'deal_dates',
              'debt_value', 'rating', 'debt_rating',
              'type', 'double_rank', 'pure_debt_prem', 'debt_rank', 'equity_rank', 'rank']

# sorted(df['turn'],reverse=Flse)[int(df.shape[0]*0.1)]
# _turn_median = df.ilaoc[:, 6].astype(float).median()
# _turn_median = df['turn'].quantile(0.25)
_turn_median = \
pd.DataFrame([float(item.strip('%')) / 100 for item in list(df['turn']) if not isinstance(item, float)]).quantile(0.25)[
    0]
#
# in_codes = [not str(item).startswith('132') for item in df["代码"]]
# df = df[in_codes]
# df = df[df.有条件赎回价 == np.inf]
# df = df[df.有条件回售价 == np.inf]
# df = df[df.剩余期限 > 1.0]
# df = df[df.最新 > 0]
# df = df[df.最新 < 115]
# df = df.loc[[float(item) > _turn_median for item in df['turn']]]
# df = df.sort_values(by='rank')

df = df.loc[[(not str(item).startswith('132')) for item in df['sec_code']]]
df = df.loc[df['cond_sale_price'] == np.inf]
df = df.loc[df['cond_repurchase_price'] == np.inf]
df = df.loc[[item > 1.0 for item in df['remain_dates'].astype(float)]]
df = df.loc[[item > 0 for item in df['last_price'].astype(float)]]
df = df.loc[[item < 120 for item in df['last_price'].astype(float)]]
_turn = [float(item.strip('%')) / 100 for item in list(df['turn']) if not isinstance(item, float)]
df['turn'] = _turn
df = df.loc[[float(item) > _turn_median for item in df['turn']]]
df = df.loc[["ST" not in item for item in df['equ_name']]]

df = df.sort_values(by='rank', ascending=True)

df.to_csv('cache/mkt_{0}_rank.csv'.format(trade_date), encoding='gbk', index=False)
