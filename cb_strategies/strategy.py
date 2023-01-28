#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/27 21:10
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : strategy.py

import pandas as pd
import pprint
import numpy as np
from model_processing.monte_carlo import MonteCarlo
from data_processing.fetch_data import get_conv_bond_statics
from data_processing.fetch_data import get_conv_bond_mkts
from data_processing.fetch_data import get_equ_mkts
from data_processing.fetch_data import get_acc_dates
from data_processing.fetch_data import get_trade_calendar
from data_processing.fetch_data import get_cb_der_mkts
from data_processing.fetch_data import get_st_flag
from data_processing.fetch_data import get_industry_name


def get_source_mkt(trade_date=''):
    conv_bond_statics, cnt = get_conv_bond_statics(start_date=trade_date, end_date=trade_date)
    sec_ids = list(set(conv_bond_statics['secID']))
    mkt_df, cnt = get_cb_der_mkts(sec_ids=sec_ids, start_date=trade_date, end_date=trade_date)
    return mkt_df


def get_portfolio(df=None, trade_date='', equ_ratio=0.7, turn_quant=0.25):
    # df['pure_debt_prem'] = df['最新'] / df['纯债价值'] - 1
    df['debt_rank'] = df['puredebtPremRatio'].rank()
    df['equity_rank'] = df['bondPremRatio'].rank()
    df['rank'] = df['debt_rank'] * (1 - equ_ratio) + df['equity_rank'] * equ_ratio
    _turn_median = df['turnoverRate'].quantile(turn_quant)
    df = df.loc[[(not str(item).startswith('132')) for item in df['ticker']]]

    # TODO 找公告事件看有没赎回触发通知
    # df = df.loc[df['putPrice'] == np.inf]
    # df = df.loc[df['callPrice'] == np.inf]

    df = df.loc[[item > 0.5 for item in df['yearToMat'].astype(float)]]
    df = df.loc[[item > 0 for item in df['closePrice'].astype(float)]]
    df = df.loc[[item < 120 for item in df['closePrice'].astype(float)]]
    df = df.loc[[item > _turn_median for item in df['turnoverRate'].astype(float)]]


    # remove ST
    st_lst = get_st_flag(start_date=trade_date, end_date=trade_date, ticker=list(df['tickerSymbolEqu']))[0]['ticker']
    df = df.loc[[item not in list(st_lst) for item in list(df['tickerSymbolEqu'])]]

    indus_df = get_industry_name(ticker=list(df['tickerSymbolEqu']))[['ticker', 'industryName1']]

    indus_dict = dict(zip(indus_df['ticker'], indus_df['industryName1']))
    _indus_lst = [indus_dict.get(k) for k in list(df['tickerSymbolEqu'])]

    df['industryName1'] = _indus_lst
    pprint.pprint(df.iloc[:20, :][['ticker', 'secShortName',  'industryName1','closePrice']])
    df.to_csv('cache/mkt_{0}_rank.csv'.format(trade_date), encoding='gbk', index=False)


def strategy_from_wind():
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

    _turn_median = \
        pd.DataFrame(
            [float(item.strip('%')) / 100 for item in list(df['turn']) if not isinstance(item, float)]).quantile(0.25)[
            0]

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


if __name__ == '__main__':
    trade_date = '20230120'
    df = get_source_mkt(trade_date=trade_date)
    get_portfolio(trade_date=trade_date, df=df, equ_ratio=0.7, turn_quant=0.25)
