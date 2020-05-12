#!/usr/bin/env python
# encoding: utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: monte_carlo.py
@time: 2020/4/23 10:04
@desc:
'''

import numpy as np
from sklearn.linear_model import LinearRegression


# https://www.investopedia.com/articles/07/montecarlo.asp
class MonteCarlo:
    def __init__(self, S0, K, T, r, q, sigma, underlying_process="geometric brownian motion"):
        self.underlying_process = underlying_process
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.sigma = sigma
        self.value_results = None

    # view antithetic variates as a option of simulation method to reduce the variance
    def simulate(self, n_trials, n_steps, antitheticVariates=False, boundaryScheme="Higham and Mao"):

        dt = self.T / n_steps
        mu = self.r - self.q
        self.n_trials = n_trials
        self.n_steps = n_steps
        self.boundaryScheme = boundaryScheme

        if self.underlying_process == "geometric brownian motion":
            #             first_step_prices = np.ones((n_trials,1))*np.log(self.S0)
            log_price_matrix = np.zeros((n_trials, n_steps))
            normal_matrix = np.random.normal(size=(n_trials, n_steps))
            if antitheticVariates:
                n_trials *= 2
                self.n_trials = n_trials
                normal_matrix = np.concatenate((normal_matrix, -normal_matrix), axis=0)
            cumsum_normal_matrix = normal_matrix.cumsum(axis=1)
            #             log_price_matrix = np.concatenate((first_step_prices,log_price_matrix),axis=1)
            deviation_matrix = cumsum_normal_matrix * self.sigma * np.sqrt(dt) + \
                               (mu - self.sigma ** 2 / 2) * dt * np.arange(1, n_steps + 1)
            log_price_matrix = deviation_matrix + np.log(self.S0)
            price_matrix = np.exp(log_price_matrix)
            price_zero = (np.ones(n_trials) * self.S0)[:, np.newaxis]
            price_matrix = np.concatenate((price_zero, price_matrix), axis=1)
            self.price_matrix = price_matrix

        elif (self.underlying_process == "CIR model"):
            # generate correlated random variables, check Monte Carlo Project
            pass
        elif (self.underlying_process == "Heston model"):
            # generate correlated random variables, check monte Carlo project
            pass
        return price_matrix

    def LSM(self, option_type="c", func_list=[lambda x: x ** 0, lambda x: x], onlyITM=False, buy_cost=0.0,
            sell_cost=0.0):
        """
        onlyITM=True: A1 strategy (i.e. LSM method from Longstaff and Schwartz)
        onlyITM=False: A2b strategy (i.e. Hedged LSM method implemented by Yuxuan Xia)
        """

        dt = self.T / self.n_steps
        df = np.exp(-self.r * dt)
        # df2 = np.exp(-(self.r - self.q) * dt)
        K = self.K
        price_matrix = self.price_matrix
        n_trials = self.n_trials
        n_steps = self.n_steps
        exercise_matrix = np.zeros(price_matrix.shape, dtype=bool)
        american_values_matrix = np.zeros(price_matrix.shape)

        def _calc_american_values(payoff_fun, func_list, sub_price_matrix, sub_exercise_matrix, df, onlyITM=False):
            exercise_values_t = payoff_fun(sub_price_matrix[:, 0])
            ITM_filter = exercise_values_t > 0
            OTM_filter = exercise_values_t <= 0
            n_sub_trials, n_sub_steps = sub_price_matrix.shape
            holding_values_t = np.zeros(n_sub_trials)  # simulated samples: y
            exp_holding_values_t = np.zeros(n_sub_trials)  # regressed results: E[y]

            itemindex = np.where(sub_exercise_matrix == 1)
            # print(sub_exercise_matrix)
            for trial_i in range(n_sub_trials):
                first = next(itemindex[1][i] for i, x in enumerate(itemindex[0]) if x == trial_i)
                payoff_i = payoff_fun(sub_price_matrix[trial_i, first])
                df_i = df ** (n_sub_steps - first)
                holding_values_t[trial_i] = payoff_i * df_i

            A_matrix = np.array([func(sub_price_matrix[:, 0]) for func in func_list]).T
            b_matrix = holding_values_t[:, np.newaxis]  # g_tau|Fi
            ITM_A_matrix = A_matrix[ITM_filter, :]
            ITM_b_matrix = b_matrix[ITM_filter, :]
            lr = LinearRegression(fit_intercept=False)
            lr.fit(ITM_A_matrix, ITM_b_matrix)
            exp_holding_values_t[ITM_filter] = np.dot(ITM_A_matrix, lr.coef_.T)[:, 0]  # E[g_tau|Fi] only ITM

            if np.sum(
                    OTM_filter):
                # if no trial falls into the OTM region it would cause empty OTM_A_Matrix and OTM_b_Matrix,
                # and only ITM was applicable. In this step, we are going to estimate the OTM American values
                # E[g_tau|Fi].
                if onlyITM:
                    # Original LSM
                    exp_holding_values_t[OTM_filter] = np.nan
                else:
                    # non-conformed approximation: do not assure the continuity of the approximation (regression
                    # in two region without iterpolation)
                    OTM_A_matrix = A_matrix[OTM_filter, :]
                    OTM_b_matrix = b_matrix[OTM_filter, :]
                    lr.fit(OTM_A_matrix, OTM_b_matrix)
                    exp_holding_values_t[OTM_filter] = np.dot(OTM_A_matrix, lr.coef_.T)[:, 0]  # E[g_tau|Fi] only OTM

            sub_exercise_matrix[:, 0] = ITM_filter & (exercise_values_t > exp_holding_values_t)
            american_values_t = np.maximum(exp_holding_values_t, exercise_values_t)
            return american_values_t

        if option_type == "c":
            payoff_fun = lambda x: np.maximum(x - K, 0)
        elif option_type == "p":
            payoff_fun = lambda x: np.maximum(K - x, 0)

        # when contract is at the maturity
        stock_prices_t = price_matrix[:, -1]
        exercise_values_t = payoff_fun(stock_prices_t)
        holding_values_t = exercise_values_t
        american_values_matrix[:, -1] = exercise_values_t
        # TODO check exercise_matrix changes
        exercise_matrix[:, -1] = 1

        # before maturaty
        # FIXME when to use this

        # for i in np.arange(n_steps)[:0:-1]:
        #     sub_price_matrix = price_matrix[:, i:]
        #     sub_exercise_matrix = exercise_matrix[:, i:]
        #     american_values_t = _calc_american_values(payoff_fun, func_list, sub_price_matrix, sub_exercise_matrix, df,
        #                                               onlyITM)
        #     american_values_matrix[:, i] = american_values_t

        # obtain the optimal policies at the inception
        holding_matrix = np.zeros(exercise_matrix.shape, dtype=bool)
        for i in np.arange(n_trials):
            exercise_row = exercise_matrix[i, :]
            if (exercise_row.any()):
                exercise_idx = np.where(exercise_row == 1)[0][0]
                exercise_row[exercise_idx + 1:] = 0
                holding_matrix[i, :exercise_idx + 1] = 1
            else:
                exercise_row[-1] = 1
                holding_matrix[i, :] = 1
        self.holding_matrix = holding_matrix
        self.exercise_matrix = exercise_matrix
        pass

    def MCPricer(self, option_type='c', isAmerican=False):
        price_matrix = self.price_matrix
        n_steps = self.n_steps
        n_trials = self.n_trials
        strike = self.K
        risk_free_rate = self.r
        time_to_maturity = self.T
        dt = time_to_maturity / n_steps
        if (option_type == "c"):
            payoff_fun = lambda x: np.maximum(x - strike, 0)
        elif (option_type == "p"):
            payoff_fun = lambda x: np.maximum(strike - x, 0)
        else:
            print("please enter the option type: (c/p)")
            return
        if not isAmerican:
            payoff = payoff_fun(price_matrix[:, n_steps])
            #         vk = payoff*df
            value_results = payoff * np.exp(-risk_free_rate * time_to_maturity)
            self.payoff = payoff
        else:
            exercise_matrix = self.exercise_matrix
            t_exercise_array = dt * np.where(exercise_matrix == 1)[1]
            value_results = payoff_fun(price_matrix[np.where(exercise_matrix == 1)]) * np.exp(
                -risk_free_rate * t_exercise_array)

        regular_mc_price = np.average(value_results)
        self.mc_price = regular_mc_price
        self.value_results = value_results
        return regular_mc_price


if __name__ == '__main__':
    risk_free_rate = 0.1
    dividend = 0
    time_to_maturity = 0.25
    volatility = 0.9
    strike = 10
    stock_price = 8
    # V0 = 0.010201
    # kappa = 6.21
    # theta = 0.019
    # xi = 0.61
    # rho = -0.7

    n_trials = 1000
    n_steps = 20
    func_list = [lambda x: x ** 0, lambda x: x]  # basis for OHMC part
    option_type = 'p'
    mc = MonteCarlo(S0=stock_price, K=strike, T=time_to_maturity, r=risk_free_rate, q=dividend, sigma=volatility,
                    underlying_process="geometric brownian motion")
    price_matrix = mc.simulate(n_trials=n_trials, n_steps=n_steps, boundaryScheme="Higham and Mao")
    print(price_matrix)
    price_matrix = np.array(price_matrix)
    mc.price_matrix = price_matrix
    # TODO  func_lst: no x**2?
    mc.LSM(option_type=option_type, func_list=[lambda x: x ** 0, lambda x: x], onlyITM=False, buy_cost=0.0,
           sell_cost=0.0)
    ret = mc.MCPricer(option_type=option_type, isAmerican=True)
    print(ret)
