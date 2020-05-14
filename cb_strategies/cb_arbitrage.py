#!/usr/bin/env python
# encoding: utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: cb_arbitrage.py
@time: 2020/4/23 15:24
@desc:
'''

import datetime
import numpy as np
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import pprint
from collections import defaultdict
from model_processing.monte_carlo import MonteCarlo
from data_processing.fetch_data import get_conv_bond_statics
from data_processing.fetch_data import get_conv_bond_mkts
from data_processing.fetch_data import get_equ_mkts
from data_processing.fetch_data import get_acc_dates
from data_processing.fetch_data import get_trade_cal
from utils.logger import Logger
from utils.helper import write_json_file

logger = Logger().get_log()


def get_conv_bond_evaluates(trade_date='20200402', start_date='20200401', end_date='20200403', risk_free_rate=0.1,
                            dividend=0):
    df_trade_cal, call_cnt = get_trade_cal(start_date, end_date)
    logger.info('complete trade cal query with call count:{0}'.format(call_cnt))
    df_trade_cal = df_trade_cal[df_trade_cal.isWeekEnd == 1]
    _dates = list(df_trade_cal['calendarDate'])
    trade_dates = list(set([item.replace('-', '') for item in _dates]))
    trade_dates = sorted(trade_dates)
    conv_bond_statics, call_cnt = get_conv_bond_statics(start_date=start_date, end_date=end_date)
    logger.info('complte bond statics query, with call count:{0}'.format(call_cnt))
    # sec_ids = ['113503.XSHG']
    sec_ids = list(set(conv_bond_statics['secID']))
    # # FIXME for testing purpose
    # sec_ids = sorted(sec_ids)[:2]
    conv_bond_mkts, call_cnt = get_conv_bond_mkts(sec_ids=sec_ids, start_date=start_date, end_date=end_date)
    logger.info('complte conv bond mkt query with call count:{0}'.format(call_cnt))
    exchange_cds = [item.split('.')[1] for item in conv_bond_mkts['secID']]
    equ_sec_ids = ['{0}.{1}'.format(conv_bond_mkts['tickerEqu'][idx], item) for idx, item in enumerate(exchange_cds)]
    conv_id2equ_id = dict(zip(conv_bond_mkts['secID'], equ_sec_ids))
    equ_mkts, call_cnt = get_equ_mkts(list(set(equ_sec_ids)), start_date, end_date)
    logger.info('complete equ mkt query with call count:{0}'.format(call_cnt))
    portfolios = defaultdict(list)
    mkt_values = dict()
    df_acc_info, call_cnt = get_acc_dates(sec_ids=sec_ids, start_date=start_date, end_date='')
    logger.info('complete acc_info query with call count:{0}'.format(call_cnt))

    all_rows = []
    for d in trade_dates:
        cb_price_evaluates = []
        logger.info('Processing trade date:{0}'.format(d))
        _format_trade_date = datetime.datetime.strptime(d, '%Y%m%d').strftime('%Y-%m-%d')
        # acc_info_dicts = get_acc_dates(sec_ids=sec_ids, trade_date=d)
        _date_cb_mkt = conv_bond_mkts[conv_bond_mkts.tradeDate == _format_trade_date]
        _date_cb_mkt.sort_values(by='puredebtPremRatio', inplace=True)
        n_rows = _date_cb_mkt.shape[0]
        _date_cb_mkt = _date_cb_mkt.head(int(n_rows / 2))

        _sec_ids = list(set(_date_cb_mkt['secID']))
        # # FIXME change to one query
        # acc_info_dicts = get_acc_dates(sec_ids=_sec_ids, trade_date=d)

        for sec_id in _sec_ids:
            logger.info('Processing sec_id :{0},date:{1}'.format(sec_id, d))
            # acc_end_date = acc_info_dicts.get(sec_id)
            _df_acc = df_acc_info[df_acc_info.secID == sec_id].sort_values(by='perAccrEndDate', ascending=False,
                                                                           inplace=False)
            # _dates = _df_acc[['perAccrDate', 'perAccrEndDate']]
            acc_end_date = list(_df_acc['perAccrEndDate'])
            # TODO check the date
            if not acc_end_date:
                continue
            _cb_mkt = _date_cb_mkt[_date_cb_mkt.secID == sec_id]
            # _cb_mkt = _cb_mkt[_cb_mkt.tradeDate == _format_trade_date]
            cb_close = list(_cb_mkt['closePriceBond'])[0]
            pure_bond_close = list(_cb_mkt['debtPuredebtRatio'])[0]
            _equ_mkt = equ_mkts.get(conv_id2equ_id.get(sec_id))
            _equ_mkt = _equ_mkt[_equ_mkt.tradeDate == _format_trade_date]
            T = (datetime.datetime.strptime(acc_end_date[0], '%Y-%m-%d') - datetime.datetime.strptime(d,
                                                                                                      '%Y%m%d')).days / 365
            risk_free_rate = risk_free_rate
            dividend = dividend
            time_to_maturity = T
            vol_lst = list(_equ_mkt['annual_vol'])
            if not vol_lst:
                logger.debug("Missing volatility value for sec_id:{0}, trade_date:{1}".format)
                continue
            volatility = list(_equ_mkt['annual_vol'])[0]
            volatility = vol_lst[0]
            strike = list(_cb_mkt['convPrice'])[0]
            stock_price = list(_equ_mkt['closePrice'])[0]

            n_trials = 1000
            n_steps = 20
            func_list = [lambda x: x ** 0, lambda x: x]  # basis for OHMC part
            option_type = 'c'

            mc = MonteCarlo(S0=stock_price, K=strike, T=time_to_maturity, r=risk_free_rate, q=dividend,
                            sigma=volatility,
                            underlying_process="geometric brownian motion")
            price_matrix = mc.simulate(n_trials=n_trials, n_steps=n_steps, boundaryScheme="Higham and Mao")
            price_matrix = np.array(price_matrix)
            mc.price_matrix = price_matrix
            # TODO  func_lst: no x**2?
            mc.LSM(option_type=option_type, func_list=func_list, onlyITM=True, buy_cost=0.0,
                   sell_cost=0.0)
            option_price = mc.MCPricer(option_type=option_type, isAmerican=True)
            evaluate_price = pure_bond_close + option_price
            # cb_price_evaluates.append(
            #     [sec_id, cb_close, evaluate_price, cb_close / evaluate_price, d, pure_bond_close, option_price, T,
            #      volatility, strike, stock_price, acc_end_date[0]])
            all_rows.append(
                [sec_id, cb_close, evaluate_price, cb_close / evaluate_price, d, pure_bond_close, option_price, T,
                 volatility, strike, stock_price, acc_end_date[0]])
        # cb_price_evaluates = sorted(cb_price_evaluates, key=lambda x: x[2])
        #
        # cb_price_evaluates = cb_price_evaluates[:topk]
        # _mkt_val = np.array([item[1] for item in cb_price_evaluates]).mean()
        # portfolios[d] = cb_price_evaluates[:topk]
        # mkt_values.update({d: _mkt_val})
    df_all_cb_evaluates = pd.DataFrame(all_rows, columns=['sec_id', 'cb_close', 'evalute_price',
                                                          'prem_ratio', 'trade_date',
                                                          'pure_bond_close', 'option_price', 'duration',
                                                          'volatality', 'strike_price', 'stock_price',
                                                          'accrual_end_date'])
    df_all_cb_evaluates.to_csv("cb_evaluates_results_{0}_{1}.csv".format(start_date, end_date), index=False)


def get_mkt_values(start_date='', end_date='', topk=20):
    df = pd.read_csv('cb_evaluates_results_{0}_{1}.csv'.format(start_date, end_date))
    all_dates = list(set(df['trade_date']))
    weekly_values = list()
    weekly_portfolios = list()
    all_dates = sorted(all_dates)
    metric_dict = dict()
    for d in all_dates:
        _df = df[df.trade_date == d]
        _df.sort_values(by='prem_ratio', ascending=True, inplace=True)
        cb_close_lst = list(_df['cb_close'])[:topk]
        sec_id_lst = list(_df['sec_id'])[:topk]
        sec_id_lst = sorted(sec_id_lst)
        sec_id_lst.append(d)
        weekly_values.append(sum(cb_close_lst))
        # weekly_values.update({d: sum(cb_close_lst)})
        weekly_portfolios.append(sec_id_lst)

    df_portfolios = pd.DataFrame(weekly_portfolios)
    all_returns = weekly_values[-1] / weekly_values[0] - 1

    annual_return = math.pow(all_returns, len(weekly_values) / 52)
    annual_vol = np.array(weekly_values).std() * math.sqrt(52)
    max_drawdown = 0.0
    idx = 0
    left, right = weekly_values[0], 0
    while idx < len(weekly_values) - 1:
        if weekly_values[idx + 1] < weekly_values[idx]:
            right = weekly_values[idx + 1]
            if abs(float(right / left) - 1) > max_drawdown:
                max_drawdown = abs(float(right / left) - 1.0)
        else:
            left = weekly_values[idx + 1]
        idx += 1
    metric_dict.update({'start_date': start_date, 'end_date': end_date, 'topk': topk, 'total_return': all_returns,
                        'annual_return': annual_return, 'annual_vol': annual_vol,
                        'max_drawdown': max_drawdown})
    write_json_file('bact_testing_metrics.json',metric_dict)
    df_portfolios.to_csv('portfolio_{0}_{1}.csv'.format(start_date, end_date))

    net_values = [item / weekly_values[0] for item in weekly_values]
    # plt.plot(net_values)
    all_dates_str = [str(item) for item in all_dates]
    plt.plot(all_dates_str, net_values)
    plt.gcf().autofmt_xdate()
    plt.savefig('weekly_values_{0}_{1}.jpg'.format(start_date, end_date))


def back_testing(start_date='', end_date='', *args, **kwargs):
    topk = kwargs.get('topk') or 20


if __name__ == '__main__':
    s1 = "20200401"
    e1 = "20200416"
    s2 = '20200103'
    e2 = '20200425'
    get_conv_bond_evaluates(trade_date='20200402', start_date=s2, end_date=e2, risk_free_rate=0.1, dividend=0,
                            topk=10)
    get_mkt_values(start_date=s2, end_date=e2, topk=20)
