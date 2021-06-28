#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/27 21:10
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : strategy.py

import pandas as pd

# df = pd.read_excel('cache/mkt20210627.xlsx', encoding='gbk')
# print(df.shape)

# with open('cache/mkt20210627.csv', 'r') as f:
#     ret = f.readlines()
#     print(ret[0])
#     print(len(ret))

df = pd.read_csv('cache/mkt20210627.csv')
print(df.columns)
df['pure_debt_prem'] = df['最新']/df['纯债价值']-1
df['debt_rank'] = df['pure_debt_prem'].rank()
df['equity_rank'] = df['转股溢价率'].rank()
df['rank'] = df['debt_rank'] + df['equity_rank']
df = df.sort_values(by='rank')
print(df.head(20))
df = df[df.最新 >0]
df.to_csv('cache/mkt_20210625_rank.csv',index=False)



