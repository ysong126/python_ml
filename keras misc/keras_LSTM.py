# following the tutorial
# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
from numpy import concatenate
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from datetime import datetime

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from matplotlib import pyplot

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# parse time into one column
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')


dataset = read_csv('raw.csv', parse_dates=[['year', 'month', 'day', 'hour']], index_col=0,date_parser=parse)
dataset.drop('No', axis=1, inplace=True)

dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'

dataset['pollution'].fillna(0, inplace=True)
dataset = dataset[24:]
print(dataset.head(5))
dataset.to_csv('pollution.csv')

# plotting the values
values = dataset.values
groups = range(8)

# subplot starting index is 1
i=1
fig = pyplot.figure()
for group in groups:
    pyplot.subplot(8, 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i+=1
#pyplot.show()


# convert time series to supervised learning
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


encoder = LabelEncoder()
# use integer to encode wind direction/ensure float type
values[:, 4] = encoder.fit_transform(values[:, 4])
values = values.astype('float32')

# normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


reframed = series_to_supervised(scaled, col_names=dataset.columns, n_in=1, dropnan=True)

# drop the columns of X, the columns that we don't predict
reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
print(reframed.head(5))

# split train / test
values = reframed.values
n_train_hours = 365*24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape to the LSTM argument shape (num of samples, time step, features)
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# LSTM network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit
history = model.fit(train_X, train_y,epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
pyplot.figure()
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# evaluate
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

# MSE
mse = mean_squared_error(inv_y, inv_yhat)
print("the MSE of LSTM is {}".format(mse))
