import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def load_data(file_name):
    df = pd.read_csv(file_name)
    return df

def buy_n_sell(true_prices, pred_prices, balance=1000000):

    num_share = 0
    portfolio_values = list()

    long_pos = 1  # 1 long ; 0 short
    long_price = true_prices[0] # assume holding the stock at the beginning

    profits = list() # daily profit

    num_share = balance//long_price
    balance = balance-num_share*long_price

    for i in range(len(pred_prices)-1):
        if i == len(pred_prices)-2 and long_pos == 1:
            if pred_prices[i+1] >= true_prices[i]:
                profits.append(true_prices[i+1]-long_price)  # sell on last day
                long_pos = 0
                balance = balance + true_prices[i + 1] * num_share
                portfolio_values.append(balance)  # 2nd last day
                portfolio_values.append(balance)  # last day
                num_share = 0
                break
            else:  # i == len(pred_prices)-2 and pred_prices[i+1] < true_prices[i] and long_pos == 1:
                profits.append(true_prices[i]-long_price)  # sell on the 2nd last day
                long_pos = 0
                balance = balance + num_share*true_prices[i]
                portfolio_values.append(balance)
                portfolio_values.append(balance)
                num_share = 0
                break
        if high_return(true_prices[i], pred_prices[i+1]) and long_pos == 0:
            long_pos = 1
            profits.append(0)
            long_price = true_prices[i]
            portfolio_values.append(balance)
            num_share = balance//long_price
            balance = balance-num_share*long_price
        elif high_return(true_prices[i], pred_prices[i+1]) != True and long_pos == 1:
            long_pos = 0
            profits.append(true_prices[i]-long_price)
            balance = balance + num_share*true_prices[i]
            num_share = 0
            portfolio_values.append(balance)
        elif low_return(true_prices[i], pred_prices[i+1]) and long_pos == 1:
            long_pos = 0
            balance += num_share*true_prices[i]
            portfolio_values.append(balance)
            num_share=0
        else:
            profits.append(0)
            portfolio_values.append(balance+num_share*true_prices[i])
            continue
    return portfolio_values



def high_return(true_price, pred_price):
    return pred_price >= true_price * 1.02 > 0  # if predicted price is 2% higher than current price


def low_return(true_price, pred_price):
    return true_price * 0.99 >= pred_price > 0


file_name = 'return_df.csv'
df = load_data(file_name)

true_prices, pred_prices = df['true_prices'], df['pred_prices']
balance = 1000000
portfolio_values = buy_n_sell(true_prices, pred_prices,balance)
portfolio_values = [elem/balance for elem in portfolio_values]
true_pct_change = [x/true_prices[0] for x in true_prices]

# plotting
plt.plot(true_pct_change, label="buy and hold")
plt.plot(portfolio_values, label='strategy')
plt.legend()
plt.show()

true_returns = np.diff(true_prices)
pred_returns = np.diff(pred_prices)
ax= plt.figure()
plt.plot(true_returns, label='true')
plt.plot(pred_returns, label='predicted')
plt.show()