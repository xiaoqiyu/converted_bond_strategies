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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_processing.monte_carlo import MonteCarlo
from data_processing.fetch_data import get_conv_bond_statics
from data_processing.fetch_data import get_conv_bond_mkts
from data_processing.fetch_data import get_equ_mkts
from data_processing.fetch_data import get_acc_dates
from data_processing.fetch_data import get_trade_cal
from data_processing.fetch_data import get_bc_mkts
from utils.logger import Logger
from utils.helper import write_json_file
from utils.metric_calculate import get_annual_returns
from utils.metric_calculate import get_total_returns
from utils.metric_calculate import get_max_drawdown
from utils.metric_calculate import get_net_values
from utils.metric_calculate import get_sharp_ratio
from utils.metric_calculate import get_annual_vol

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
    conv_bond_mkts, call_cnt = get_conv_bond_mkts(sec_ids=sec_ids, start_date=start_date, end_date=end_date)
    logger.info('complte conv bond mkt query with call count:{0}'.format(call_cnt))
    exchange_cds = [item.split('.')[1] for item in conv_bond_mkts['secID']]
    equ_sec_ids = ['{0}.{1}'.format(conv_bond_mkts['tickerEqu'][idx], item) for idx, item in enumerate(exchange_cds)]
    conv_id2equ_id = dict(zip(conv_bond_mkts['secID'], equ_sec_ids))
    equ_mkts, call_cnt = get_equ_mkts(list(set(equ_sec_ids)), start_date, end_date)
    logger.info('complete equ mkt query with call count:{0}'.format(call_cnt))
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
            # logger.info('Processing sec_id :{0},date:{1}'.format(sec_id, d))
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
            all_rows.append(
                [sec_id, cb_close, evaluate_price, cb_close / evaluate_price, d, pure_bond_close, option_price, T,
                 volatility, strike, stock_price, acc_end_date[0]])
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
        weekly_portfolios.append(sec_id_lst)

    # back testing metrics calcualtion for portfolio
    df_portfolios = pd.DataFrame(weekly_portfolios)
    total_returns = get_total_returns(weekly_values)
    annual_return = get_annual_returns(weekly_values)
    annual_vol = get_annual_vol(weekly_values)
    sharp_ratio = round(annual_return / annual_vol, 4)
    max_drawdown, max_idx, min_idx = get_max_drawdown(weekly_values)
    metric_dict.update(
        {'start_date': start_date, 'end_date': end_date, 'topk': topk, 'total_return': total_returns,
         'annual_return': annual_return, 'annual_vol': annual_vol,
         'max_drawdown': max_drawdown, 'sharp_ratio': sharp_ratio})
    write_json_file('back_testing_metrics_{0}_{1}.json'.format(start_date, end_date), metric_dict)
    df_portfolios.to_csv('portfolio_{0}_{1}.csv'.format(start_date, end_date))
    net_values = get_net_values(weekly_values)
    all_dates_str = [str(item) for item in all_dates]
    n_dates = len(all_dates)

    # fetch and calculate zzcb benchmark
    df_bc_mkts, call_cnt = get_bc_mkts(start_date=all_dates_str[0], end_date=all_dates_str[-1], ticker='000832')
    logger.info('complte conv_bond benchmark mkts query with call count:{0}'.format(call_cnt))
    bc_trade_dates = [item.replace('-', '') for item in df_bc_mkts['tradeDate']]
    bc_values = []
    close_indexs = list(df_bc_mkts['closeIndex'])
    for d in all_dates_str:
        _val = close_indexs[bc_trade_dates.index(d)]
        bc_values.append(_val)
    bc_net_values = get_net_values(bc_values)

    # fetch and calculate zz500 benchmark
    df_zz500_mkts, call_cnt = get_bc_mkts(start_date=all_dates_str[0], end_date=all_dates_str[-1], ticker='000905')
    logger.info('complte zz500 benchmark mkts query with call count:{0}'.format(call_cnt))
    bc_zz500_trade_dates = [item.replace('-', '') for item in df_bc_mkts['tradeDate']]
    bc_zz500_values = []
    close_indexs_zz500 = list(df_zz500_mkts['closeIndex'])
    for d in all_dates_str:
        _val = close_indexs_zz500[bc_zz500_trade_dates.index(d)]
        bc_zz500_values.append(_val)
    bc_zz500_net_values = get_net_values(bc_zz500_values)

    x_tickers = []
    idx = 0
    while idx < n_dates:
        x_tickers.append(idx)
        idx += 4
    x_labels = [all_dates[idx] for idx in x_tickers]
    plt.xticks(x_tickers, x_labels)
    plt.annotate("({0},{1})".format(all_dates[max_idx], net_values[max_idx]), xytext=(max_idx, net_values[max_idx]),
                 xy=(max_idx, net_values[max_idx]))
    plt.annotate("({0},{1})".format(all_dates[min_idx], net_values[min_idx]), xytext=(min_idx, net_values[min_idx]),
                 xy=(min_idx, net_values[min_idx]))
    plt.plot(all_dates_str, net_values)
    plt.plot(all_dates_str, bc_net_values)
    plt.plot(all_dates_str, bc_zz500_net_values)

    plt.legend(['portfolio', 'zzconv_bond', 'zz500'])
    plt.gcf().autofmt_xdate()
    plt.title("total_return:{0}  annual_return:{1}  max_drawdown:{2}%, topk:{3}".format(metric_dict.get('total_return'),
                                                                                        metric_dict.get(
                                                                                            'annual_return'),
                                                                                        metric_dict.get(
                                                                                            'max_drawdown') * 100,
                                                                                        metric_dict.get('topk')))
    # plt.show()
    plt.savefig('weekly_values_{0}_{1}.jpg'.format(start_date, end_date))


def back_testing(start_date='', end_date='', **kwargs):
    topk = kwargs.get('topk') or 20
    get_conv_bond_evaluates(trade_date='20200402', start_date=start_date, end_date=end_date, risk_free_rate=0.1,
                            dividend=0)
    get_mkt_values(start_date=start_date, end_date=end_date, topk=topk)


if __name__ == '__main__':
    s1 = "20190103"
    e1 = "20200416"
    s2 = '20200103'
    e2 = '20200425'
    # get_conv_bond_evaluates(trade_date='20200402', start_date=s2, end_date=e2, risk_free_rate=0.1, dividend=0)
    get_mkt_values(start_date=s1, end_date=e2, topk=20)
    # back_testing(start_date=s1, end_date=e2 , topk=20)
