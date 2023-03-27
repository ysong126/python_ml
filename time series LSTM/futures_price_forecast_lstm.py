'''

Corn Futures price prediction using LSTM models

data: Corn futures price from Nasdaq.com

'''

import numpy as np
from numpy import concatenate

from pandas import read_csv
from pandas import concat
from pandas import DataFrame

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from matplotlib import pyplot

# load data. skip spaces in columns
corn_data_full = read_csv("HistoricalQuotesCornNasdaq.csv", sep=r'\s*,\s*')
print(corn_data_full.head(5))

# couldn't index 'Open'. the following line checks the String
# print(corn_data_full.columns.tolist())

# reverse time
corn_data = corn_data_full.iloc[::-1]

# check missing value
# null_data = corn_data[corn_data.isnull().any(axis=1)]
# null_data.shape

# drop 235 rows out of 2527 rows
corn_data.dropna(inplace=True)

# add a momentum column
corn_data['Momentum'] = (corn_data['Open']+corn_data['Close'])/2*corn_data['Volume']
print(corn_data.head(5))

# set date as index
corn_data.info()
corn_data.set_index('Date', inplace=True)

# scale data. This scaler takes 6 columns
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_corn_data = scaler.fit_transform(corn_data)

# a function that adds past values as features


def series_to_supervised(data, col_names, n_in=1, dropnan=True ):
    """
    reshape time series dataframe into an aggragated dataframe with
    past values as features.

    :param data: dataframe
    :param col_names: list of column names
    :param n_in: lagged period
    :param dropnan: drop NA values. default to True
    :return: aggregated dataframe with past values as features/columns
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # shifted dataframe put horizontally
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += ['%s(t-%d)' % (col_name, i) for col_name in col_names]
    # concatenate and drop the invalid values due to shift
    cols.append(df)
    names += ['%s(t0)' % (col_name) for col_name in col_names]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# add past values as features
n_lag = 7
n_features = 6

corn_data_supervised = series_to_supervised(scaled_corn_data, corn_data.columns.tolist(), n_in=n_lag, dropnan=True)
print(corn_data_supervised.head(5))

# drop the columns that we don't predict
corn_data_supervised = corn_data_supervised.iloc[:, :-5]

# calculate the dimension
n_obs = n_lag*n_features

# split train/test  0.9 / 0.1
n_train = int(corn_data.shape[0]*0.9)

train_data = corn_data_supervised.values[:n_train, :]
test_data = corn_data_supervised.values[n_train:, :]



train_X, train_y = train_data[:, :-1], train_data[:, -1]
test_X, test_y = test_data[:, :-1], test_data[:, -1]

# time step == n_lag; n_features == total number of columns(y included)
train_X = train_X.reshape((train_X.shape[0], n_lag, n_features))
test_X = test_X.reshape((test_X.shape[0], n_lag, n_features))

# define our LSTM architecture
# 1 vanilla/stacked
# rule of thumb: neuron number = number of Obs/(2*(num of input+ num of output ))

model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=False))
model.add(Dense(1))
model.compile(optimizer="SGD", loss="mse")

# train
history = model.fit(train_X, train_y, epochs=20, batch_size=50, validation_data=(test_X, test_y), verbose=2, shuffle = False)
pyplot.figure()
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.legend()
pyplot.show()

# evaluate
y_hat = model.predict(test_X)

# invert scaling for forecast
# first n_lag rows get dropped as na
test_X_temp = scaled_corn_data[n_train+n_lag:, 1:]
inv_yhat = concatenate((y_hat, test_X_temp), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]

test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X_temp), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

# MSE
mse = mean_squared_error(inv_y, inv_yhat)
print("the MSE of LSTM is {}".format(mse))