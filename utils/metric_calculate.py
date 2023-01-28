#!/usr/bin/env python
# encoding: utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: metric_calculate.py
@time: 2020/5/17 11:34
@desc:
'''

import math
import numpy as np
from sklearn.linear_model import LinearRegression


def get_net_values(vals):
    if not vals:
        return vals
    return [round(item / vals[0], 4) for item in vals]


def get_total_returns(vals):
    if not vals:
        return 0.0
    return round(vals[-1] / vals[0] - 1, 4)


def get_annual_returns(vals):
    total_returns = get_total_returns(vals)
    if total_returns:
        return round(total_returns / len(vals) * 52, 4)
        # return round(math.pow(total_returns, len(vals) / 52),4)


def get_sharp_ratio(vals=[], risk_free=0.01):
    annual_ret = get_annual_returns(vals)
    _vol = np.array(vals).std()
    if annual_ret and _vol:
        return round((annual_ret - risk_free) / _vol, 4)


def get_max_drawdown(vals):
    max_drawdown = 0.0
    idx = 0
    left, right = vals[0], 0
    left_idx, right_idx = 0, 0
    max_idx, min_idx = 0, 0
    while idx < len(vals) - 1:
        if vals[idx + 1] < vals[idx]:
            right = vals[idx + 1]
            right_idx = idx + 1
            if abs(float(right / left) - 1) > max_drawdown:
                max_drawdown = abs(float(right / left) - 1.0)
                right_idx = idx + 1
                max_idx = left_idx
                min_idx = right_idx
        else:
            left = vals[idx + 1]
            left_idx = idx + 1
        idx += 1
    return round(max_drawdown, 4), max_idx, min_idx


def get_annual_vol(vals):
    if not vals:
        return
    n_len = len(vals)
    arr = []
    for i in range(1, n_len):
        arr.append(vals[i] / vals[i - 1] - 1)
    return round(np.array(arr).std() * math.sqrt(52), 4)


def get_alpha_beta(vals, bc_vals):
    lr = LinearRegression(fit_intercept=True)
    X = np.array(bc_vals).reshape((-1, 1))
    y = np.array(vals).reshape((-1, 1))
    lr.fit(X, y)
    return lr.coef_[0][0], lr.intercept_[0]


if __name__ == '__main__':
    import numpy as np
    # from sklearn.linear_model import LinearRegression

    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = 1 * x_0 + 2 * x_1 + 3
    y = np.dot(X, np.array([1, 2])) + 3
    X = np.array(range(10)).reshape((-1, 1))
    y = X * 2.0 + 3
    beta, alpha = get_alpha_beta(y, X)
    print(beta)
    print(alpha)
