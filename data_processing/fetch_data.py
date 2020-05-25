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
import datetime
import pandas as pd
from uqer import DataAPI
from utils.helper import func_count
from utils.helper import timeit
from utils.logger import Logger
from account_info.get_account_info import get_account_info

# Add token to fetch the data if retrieve the data online
uqer_client = uqer.Client(token=get_account_info().get('uqer_token'))
logger = Logger().get_log()


@func_count
@timeit
def get_trade_cal(start_date='20200103', end_date='20200424'):
    return DataAPI.TradeCalGet(exchangeCD=u"XSHG,XSHE", beginDate=start_date, endDate=end_date, isOpen=u"1",
                               field=u"", pandas="1")


@func_count
@timeit
def get_conv_bond_statics(ticker="", trade_date='20200401', start_date='20200401', end_date='20200403'):
    # Fetch the conv bond ids
    # df = DataAPI.SecIDGet(partyID=u"", ticker=u"", cnSpell=u"", assetClass=u"B", field=u"", pandas="1",
    #                       listStatusCD=u"L")
    # df = df[df['secShortName'].str.contains("转债")].sort_values(by='listDate')
    df = pd.read_csv('conv_bond_codes.csv')
    sec_ids = list(set(df['secID'].values))
    fields = ['secID', 'totalSize', 'remainSize', 'firstAccrDate', 'maturityDate', 'firstRedempDate', 'publishDate',
              'listDate', 'delistDate']
    conv_bond_statics = DataAPI.BondGet(secID=sec_ids, ticker=u"", typeID=u"02020113", exchangeCD=u"", partyID=u"",
                                        listStatusCD=u"", delistDate="", field=fields, pandas="1")
    conv_bond_statics = conv_bond_statics[conv_bond_statics.listDate <= start_date]
    conv_bond_statics = conv_bond_statics[conv_bond_statics.maturityDate > end_date]
    conv_bond_statics = conv_bond_statics[conv_bond_statics.delistDate > end_date]
    return conv_bond_statics
    # return ['113503.XSHG']


@func_count
@timeit
def _get_conv_price(sec_id='', trade_date=''):
    df = DataAPI.BondConvPriceChgGet(secID=sec_id, ticker=u"", field=u"", pandas="1")
    # _dates = [item.replace('-', '') for item in df['convDate']]
    # df['convDate'] = _dates
    df = df[df.convDate <= trade_date]
    df.sort_values(by='convDate', ascending=True, inplace=True)
    return list(df['convPrice'])[-1]


@func_count
@timeit
def get_conv_bond_mkts(sec_ids=[], start_date='20200401', end_date='20200403'):
    mkt_df = DataAPI.MktConsBondPerfGet(beginDate=start_date, endDate=end_date, secID=sec_ids)
    _vals = mkt_df[['secID', 'tradeDate']].values
    # conv_prices = []
    # for sec_id, trade_date in _vals:
    #     conv_prices.append(_get_conv_price(sec_id, trade_date))
    # mkt_df['convPrice'] = conv_prices
    # # 与债底位置的距离指标
    # mkt_df['bondPosIndicator'] = ((mkt_df['closePriceBond'] - mkt_df['debtPuredebtRatio']) / mkt_df[
    #     'debtPuredebtRatio']) * 100
    return mkt_df


@func_count
@timeit
def get_equ_mkts(sec_ids=[], start_date='20200401', end_date='20200403'):
    query_start_date = datetime.datetime.strptime(start_date, '%Y%m%d') + datetime.timedelta(days=-500)
    equ_df = DataAPI.MktEqudGet(secID=sec_ids, beginDate=query_start_date.strftime('%Y%m%d'), endDate=end_date,
                                pandas="1", isOpen="1")
    ret = {}
    for sec_id in sec_ids:
        if sec_id == '300197.XSHE':
            print('check')
        try:
            _df = equ_df[equ_df.secID == sec_id]
            _df.sort_values(by='tradeDate', ascending=True, inplace=True)
            _df['annual_vol'] = _df[['chgPct']].rolling(250).apply(lambda x: x.std() * math.sqrt(250))
            _format_start_date = datetime.datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d')
            _df = _df[_df.tradeDate >= _format_start_date]
            ret.update({sec_id: _df})
        except Exception as ex:
            print("check", ex)
    return ret


@func_count
@timeit
def get_conv_price(sec_ids=[]):
    df = DataAPI.BondConvPriceChgGet(secID=sec_ids[0], ticker=u"", field=u"", pandas="1")
    for sec_id in sec_ids[1:]:
        df.append(DataAPI.BondConvPriceChgGet(secID=sec_id, ticker=u"", field=u"", pandas="1"))
    return df


@func_count
@timeit
def get_acc_dates(sec_ids=[], trade_date='', start_date='', end_date=''):
    # TODO  start_date use the early one
    acc_infos = DataAPI.BondCfGet(secID=sec_ids, ticker=u"", beginDate='20000401', endDate=end_date, cashTypeCD=u"",
                                  field=u"",
                                  pandas="1")
    return acc_infos
    # acc_infos.sort_values(by='perAccrEndDate', inplace=True)
    # acc_infos = acc_infos.groupby('secID').agg({'perAccrEndDate': ['first']})
    # return dict(zip(list(acc_infos.index), list(acc_infos.values)))


@func_count
@timeit
def get_bc_mkts(start_date='', end_date='', ticker='000832'):
    '''

    :param start_date:
    :param end_date:
    :param ticker: 000832 中证转债；000905中证500;000300沪深300
    :return:
    '''
    return DataAPI.MktIdxdGet(indexID=u"", ticker=ticker, tradeDate=u"", beginDate=start_date, endDate=end_date,
                              exchangeCD=u"XSHE,XSHG", field=u"", pandas="1")


if __name__ == "__main__":
    # df = DataAPI.BondConvPriceChgGet(secID="113503.XSHG", ticker=u"", field=u"", pandas="1")
    # print(df)
    start_date = '20200401'
    end_date = '20200403'
    # 113503
    conv_bond_statics = get_conv_bond_statics(start_date=start_date, end_date=end_date)

    # sec_ids = ['113503.XSHG']
    sec_ids = list(set(conv_bond_statics['secID']))
    sec_ids = sorted(sec_ids)[:2]
    # perAccrEndDate is the end day of current period,for each trade_date, begin_date in acc_infos should be trade_date
    # acc_infos = DataAPI.BondCfGet(secID=sec_ids, ticker=u"", beginDate=start_date, endDate=u"", cashTypeCD=u"",
    #                               field=u"",
    #                               pandas="1")
    # acc_infos.sort_values(by='perAccrEndDate', inplace=True)
    # acc_infos = acc_infos.groupby('secID').agg({'perAccrEndDate': ['first']})
    # acc_info_dicts = dict(zip(list(acc_infos.index), list(acc_infos.values)))
    acc_info_dicts = get_acc_dates(sec_ids=sec_ids, trade_date=start_date)
    df = get_conv_bond_mkts(sec_ids=sec_ids, start_date=start_date, end_date=end_date)
    # print(df.head())
    exchange_cds = [item.split('.')[1] for item in df['secID']]
    equ_sec_ids = list(set(['{0}.{1}'.format(df['tickerEqu'][idx], item) for idx, item in enumerate(exchange_cds)]))
    if not equ_sec_ids:
        print('check')
    equ_mkt = get_equ_mkts(equ_sec_ids, start_date, end_date)
    print(equ_mkt)
    # conv_price = get_conv_price(sec_ids)
    # print(conv_price)
