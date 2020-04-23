#!/usr/bin/env python
# encoding: utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: fetch_data.py
@time: 2020/4/23 15:06
@desc:
'''

import math
import uqer
import pprint
from uqer import DataAPI

uqer_client = uqer.Client(token="26356c6121e2766186977ec49253bf1ec4550ee901c983d9a9bff32f59e6a6fb")


def get_conv_bond_statics(ticker="", trade_date='20200401'):
    df = DataAPI.SecIDGet(partyID=u"", ticker=u"", cnSpell=u"", assetClass=u"B", field=u"", pandas="1",
                          listStatusCD=u"L")
    return df[df['secShortName'].str.contains("转债")].sort_values(by='listDate')
    # return ['113503.XSHG']


def get_conv_bond_mkts(sec_ids=[], start_date='20200401', end_date='20200403'):
    mkt_df = DataAPI.MktConsBondPerfGet(beginDate=start_date, endDate=end_date, secID=sec_ids)
    return mkt_df


def get_equ_mkts(sec_ids=[], start_date='20200401', end_date='20200403'):
    equ_df = DataAPI.MktEqudGet(secID=sec_ids, beginDate=start_date, endDate=end_date, pandas="1", isOpen="1")
    ret = {}
    for sec_id in sec_ids:
        _df = equ_df[equ_df.secID == sec_id]
        _df.sort_values(by='tradeDate', ascending=True, inplace=True)
        _df['annual_vol'] = _df[['chgPct']].rolling(250).apply(lambda x: x.std() * math.sqrt(250))
        ret.update({sec_id: _df})
    return ret


if __name__ == "__main__":
    start_date = '20200401'
    end_date = '20200403'
    # 113503
    conv_bond_statics = get_conv_bond_statics(ticker="")
    sec_ids = conv_bond_statics['secID'].values
    # sec_ids = ['113503.XSHG']
    df = get_conv_bond_mkts(sec_ids=sec_ids, start_date=start_date, end_date=end_date)
    print(df.head())
    exchange_cds = [item.split('.')[1] for item in df['secID']]
    equ_sec_ids = list(set(['{0}.{1}'.format(df['tickerEqu'][idx], item) for idx, item in enumerate(exchange_cds)]))
    equ_mkt = get_equ_mkts(equ_sec_ids, '20190103', end_date)
    print(equ_mkt)
