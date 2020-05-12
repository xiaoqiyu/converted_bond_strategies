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
import pprint
from collections import defaultdict
from model_processing.monte_carlo import MonteCarlo
from data_processing.fetch_data import get_conv_bond_statics
from data_processing.fetch_data import get_conv_bond_mkts
from data_processing.fetch_data import get_equ_mkts
from data_processing.fetch_data import get_acc_dates


def get_conv_pricing(trade_date='20200402', start_date='20200401', end_date='20200403', risk_free_rate=0.1, dividend=0,
                     topk=10):
    conv_bond_statics = get_conv_bond_statics(start_date=start_date, end_date=end_date)
    # sec_ids = ['113503.XSHG']
    sec_ids = list(set(conv_bond_statics['secID']))
    # # FIXME for testing purpose
    # sec_ids = sorted(sec_ids)[:2]
    conv_bond_mkts = get_conv_bond_mkts(sec_ids=sec_ids, start_date=start_date, end_date=end_date)
    exchange_cds = [item.split('.')[1] for item in conv_bond_mkts['secID']]
    equ_sec_ids = ['{0}.{1}'.format(conv_bond_mkts['tickerEqu'][idx], item) for idx, item in enumerate(exchange_cds)]
    conv_id2equ_id = dict(zip(conv_bond_mkts['secID'], equ_sec_ids))
    equ_mkts = get_equ_mkts(list(set(equ_sec_ids)), start_date, end_date)
    portfolios = defaultdict(list)
    mkt_values = dict()
    for d in [trade_date]:
        _format_trade_date = datetime.datetime.strptime(d, '%Y%m%d').strftime('%Y-%m-%d')
        # acc_info_dicts = get_acc_dates(sec_ids=sec_ids, trade_date=d)
        _date_cb_mkt = conv_bond_mkts[conv_bond_mkts.tradeDate == _format_trade_date]
        _date_cb_mkt.sort_values(by='bondPosIndicator', inplace=True)
        n_rows = _date_cb_mkt.shape[0]
        _date_cb_mkt = _date_cb_mkt.head(int(n_rows/2))
        cb_price_evaluates = []
        _sec_ids = list(set(_date_cb_mkt['secID']))
        acc_info_dicts = get_acc_dates(sec_ids=_sec_ids, trade_date=d)
        for sec_id in _sec_ids:
            acc_end_date = acc_info_dicts.get(sec_id)
            if not acc_end_date:
                continue
            _cb_mkt = _date_cb_mkt[_date_cb_mkt.secID == sec_id]
            # _cb_mkt = _cb_mkt[_cb_mkt.tradeDate == _format_trade_date]
            cb_close = list(_cb_mkt['closePriceBond'])[0]
            pure_bond_close = list(_cb_mkt['debtPuredebtRatio'])[0]
            _equ_mkt = equ_mkts.get(conv_id2equ_id.get(sec_id))
            _equ_mkt = _equ_mkt[_equ_mkt.tradeDate == _format_trade_date]
            T = (datetime.datetime.strptime(acc_end_date[0], '%Y-%m-%d') - datetime.datetime.strptime(d,
                                                                                                      '%Y%m%d')).days / 360
            risk_free_rate = risk_free_rate
            dividend = dividend
            time_to_maturity = T
            volatility = list(_equ_mkt['annual_vol'])[0]
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
            cb_price_evaluates.append([sec_id, cb_close, evaluate_price, cb_close / evaluate_price])
        cb_price_evaluates = sorted(cb_price_evaluates, key=lambda x: x[2])[:topk]
        _mkt_val = np.array([item[1] for item in cb_price_evaluates]).mean()
        portfolios[d] = cb_price_evaluates[:topk]
        mkt_values.update({d: _mkt_val})
    print(mkt_values)
    pprint.pprint(portfolios)


if __name__ == '__main__':
    get_conv_pricing()
