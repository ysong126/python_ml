'''

Wheat Futures(ZW) price prediction using LSTM models

data: ZW/CT futures price from www.macrotrends.net

Parameter Tuning

config for LSTM

[num_unit1, num_unit2, dropout1, dropout2, act,   measure]

config dict
[config -> mse]

# CT :  96 96 0.2 0.1 RMSE=5.0759 add more units
# ZW : 96 256 __  __  RMSE=25.32
'''

import numpy as np
import statistics
from numpy import concatenate

from pandas import read_csv
from pandas import concat
from pandas import DataFrame

from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM

import matplotlib.pyplot as plt

# setup random seed
np.random.seed(25)

# load data. skip spaces in columns
# ZW.csv  wheat
# CT.csv  cotton


def load_data(file_name, rev=False):
    data_frame = read_csv(file_name, sep=r'\s*,\s*')
    print(data_frame.head(5))
    null_data = data_frame[data_frame.isnull().any(axis=1)]
    print("null data dimensions:{}".format(null_data.shape))
    if "Adj Close" in data_frame.columns:
        del data_frame['Adj Close']
    if rev:
        data_frame = data_frame.iloc[::-1]
    data_frame = data_frame[data_frame['Volume'] != 0]
    data_frame.replace([np.inf, -np.inf], np.nan)
    data_frame.dropna(inplace=True)
    data_frame['Momentum'] = (data_frame['Close'] + data_frame['Open']) / 2 * data_frame['Volume']
    data_frame.set_index('Date', inplace=True)
    data_frame = data_frame[['Close', 'Open', 'Low', 'High', 'Volume', 'Momentum']]  # re-order columns
    return data_frame


def scale(dataset):
    # scale data. This scaler takes 6 columns
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data, scaler


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
    names += ['%s(t0)' % col_name for col_name in col_names]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def model_config():
    n1 = [96]
    n2 = [128]
    d1 = [0.1]
    d2 = [0.1]
    configs = list()
    for i in n1:
        for j in n2:
            for k in d1:
                for h in d2:
                    config = [i, j, k, h]
                    configs.append(config)
    return configs


def split(data_supervised, n_train=7):

    train_data = data_supervised.values[:n_train, :]
    test_data = data_supervised.values[n_train:, :]

    train_X, train_y = train_data[:, :-1], train_data[:, -1]
    test_X, test_y = test_data[:, :-1], test_data[:, -1]

    # time step == n_lag; n_features == total number of columns(y included)
    train_X = train_X.reshape((train_X.shape[0], n_lag, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_lag, n_features))

    return train_X, test_X, train_y, test_y


# LSTM architecture
def lstm( X_shape1, X_shape2, num_units1=96, num_units2=128, droprate1=0.1, droprate2=0.1, n_output=1):
    model = Sequential()
    model.add(LSTM(num_units1, activation='relu', input_shape=(X_shape1, X_shape2), return_sequences=True))
    model.add(Dropout(droprate1))
    model.add(LSTM(num_units2, activation='relu', return_sequences=False))
    model.add(Dropout(droprate2))
    model.add(Activation('sigmoid'))
    model.add(Dense(n_output)) # predicted sequence length, default to be 1 as single value
    model.compile(optimizer="adam", loss="mse")
    return model


# return sequence of predicted y of length = test_X
def evaluate(model, test_X, test_y, scaler):
    y_hat = model.predict(test_X)
    # invert scaling for prediction. First n_lag rows get dropped as na
    # n_train = int(futures_data_supervised.shape[0] * split_rate)
    test_X_temp = scaled_futures_data[n_train+n_lag:, 1:]
    inv_yhat = concatenate((y_hat, test_X_temp), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # inverse scale true y values
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X_temp), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # MSE/RMSE
    mse = mean_squared_error(inv_y, inv_yhat)
    # print("the MSE of LSTM is {}".format(mse))
    # print("the RMSE of stacked LSTM is {}".format(mse**0.5))
    # print("the Mean of forecasted price is {}".format(statistics.mean(raw_data['Close'].values)))
    return mse**0.5  # RMSE


def model_fit(data, config, n_train):
    predictions = list()
    train_X, test_X, train_y, test_y = split(data, n_train)
    n1, n2, d1, d2 = config
    model = lstm(train_X.shape[1], train_X.shape[2], n1, n2, d1, d2)
    history = model.fit(train_X, train_y, epochs=10, batch_size=50, validation_data=(test_X, test_y), verbose=0,
                       shuffle=False)
    rmse = evaluate(model, test_X, test_y, scaler)
    #print("model fit:{}".format(rmse))
    return rmse


def rolling(scaled_data, n_train_start):
    n_train = n_train_start # 500
    prediction = list()
    true_y =list()
    train_X, test_X, train_y, test_y = split(scaled_data, n_train_start)
    # 500*42 500*42, 500*1, 500*1
    model = lstm(train_X.shape[1], train_X.shape[2], 96, 128, 0, 0.1)
    # len(scaled_data) - n_train_start

    prediction_len = 50
    for i in range(prediction_len):
        print("the {}th iteration".format(i))
        train_X, test_X, train_y, test_y = split(scaled_data, n_train) #500
        test_X_sample = test_X[:1] # next sample in test X
        model.fit(train_X, train_y, epochs=1, batch_size=1, verbose=0, shuffle=False)
        y_hat = model.predict(test_X_sample)

        prediction.append(y_hat)
        true_y.append(test_y[:1])
        n_train += 1

    pred = np.array(prediction)
    pred = pred.reshape((prediction_len,))  # shape of predicted y sequence

    true_y = np.array(true_y)
    true_y = true_y.reshape((prediction_len,))  # shape of true y sequence

    # inverse scale
    X_padding = scaled_futures_data[:len(pred), :-1]
    pred=pred.reshape((len(pred), 1))
    inv_yhat = concatenate((pred, X_padding), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    true_y = true_y.reshape((len(true_y), 1))
    inv_true_y = concatenate((true_y, X_padding), axis=1)
    inv_true_y = scaler.inverse_transform(inv_true_y)
    inv_true_y = inv_true_y[:, 0]

    print("true y shape{}".format(inv_true_y.shape))
    print("pred shape{}".format(inv_yhat.shape))
    print(inv_true_y)
    print(inv_yhat)
    plt.plot(inv_true_y, label="true value")
    plt.plot(inv_yhat, label="predicted value")
    plt.ylim(0, 600)
    plt.legend()
    plt.show()
    mape = mean_absolute_percentage_error(inv_true_y, inv_yhat)
    return mape

def repeat_evaluate(data, config, n_train, n_repeat=10):
    key = str(config)
    scores_list = [model_fit(data, config, n_train) for i in range(n_repeat)]
    result = np.mean(scores_list)
    print("Model {}. RMSE:{}".format(key, result))
    return key, result


def grid_search(data, cfg_list, n_train, n_repeat):
    result = [repeat_evaluate(data, cfg, n_train, n_repeat) for cfg in cfg_list]
    result.sort(key=lambda tup: tup[1])
    return result

# run
file_name = "ZW.csv"
raw_data = load_data(file_name, False)
scaled_futures_data, scaler = scale(raw_data)

# add past values as features
n_lag = 7
n_features = 6
n_train = 7
futures_data_supervised = series_to_supervised(scaled_futures_data, raw_data.columns.tolist(), n_in=n_lag, dropnan=True)
print(futures_data_supervised.head(5))

# drop the columns that we don't predict
futures_data_supervised = futures_data_supervised.iloc[:, :-5]


mape = rolling(futures_data_supervised, 100)
print("mape is {}".format(mape))
# grid search for optimal config
# configs = model_config()
# scores = grid_search(futures_data_supervised, configs, n_train, repeat_times)


'''
# plot
pyplot.figure()
pyplot.plot(result.history['loss'], label='train')
pyplot.plot(result.history['val_loss'], label='validation')
pyplot.legend()
pyplot.show()


'''
