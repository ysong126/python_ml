import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.api import VAR

# read data
data_full = pd.read_csv("CT.csv", sep=r'\s*,\s*')
print(data_full.head(5))

# drop invalid rows and unused columns
data_full.dropna(inplace=True)
del data_full['Adj Close']

data_full = data_full[data_full['Volume'] != 0]

# set date as index
data_full.set_index('Date', inplace=True)

# create a momentum column
futures_data = data_full
#futures_data['Momentum'] = (futures_data['Close']-futures_data['Open'])/2*futures_data['Volume']

# VAR part ad fuller test
print('Aug Dicky fuller test \n' )
for c in data_full.columns:
    ad_result=adfuller(data_full[c])
    print(ad_result)

print('\n')
# co-integration test
print('Co - integration results: \n')
for i in range(5):
    for j in range(i+1, 5):
        if(i!=j):
            coint_result = coint(data_full.iloc[:,i],data_full.iloc[:,j])
            print(coint_result)

# peek
futures_data.info()

# remove inf
futures_data.replace([np.inf, -np.inf], np.nan)
futures_data.dropna(inplace=True)

# make stationary log difference and drop divided by zero rows
var_data = np.log(futures_data).diff().dropna()
var_data = var_data[['Close', 'Open', 'High', 'Low', 'Volume']]



# model
var_model = VAR(var_data)

# lag order   AIC:12, BIC:4, HQIC:6
lag_order = var_model.select_order()
print(lag_order.summary())

# model fit  BIC:4
var_result = var_model.fit(4)
print(var_result.summary())
