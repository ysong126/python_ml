import numpy as np
from pandas import read_csv, DataFrame
from pandas import datetime, to_datetime
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from sklearn.metrics import mean_absolute_error

date_parser_func = lambda x: datetime.strptime(x, "%Y-%m-%d")


def read_data(file_name):
    data_series = read_csv(file_name, sep=',', names=["date", "gdp"], header=1, parse_dates=[0], date_parser=date_parser_func)
    data_series.set_index('date', inplace=True)
    return data_series


def plot_series(data_series):
    print(data_series.head(5))
    data_series.plot(y='gdp', use_index=True, x='date')
    pyplot.show(block=False)
    return


def check_stationary(series, window_size):
    # stats
    rolling_mean = series.rolling(window=window_size).mean()
    rolling_std = series.rolling(window=window_size).std()

    # plot
    pyplot.plot(series, color='blue', label='Original')
    pyplot.plot(rolling_mean, color='red', label='Rolling Mean')
    pyplot.plot(rolling_std, color='black', label='Rolling Std')
    pyplot.legend(loc='best')
    pyplot.title('Rolling Mean and Standard Deviation')
    pyplot.show(block=False)

    # stationarity test
    adf_result = adfuller(series, maxlag=None, autolag="AIC")
    print('ADF Statistics:{}'.format(adf_result[0]))
    print('p value: {}'.format(adf_result[1]))
    print('Critical Values: ')
    for key, value in adf_result[4].items():
        print('\t{}" {}'.format(key, value))
    return


def plot_acf_pacf(series):
    f = pyplot.figure()
    ax1 = f.add_subplot(211)
    plot_pacf(series, lags=20, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_acf(series, lags=20, ax=ax2)
    pyplot.show()
    return


file_name = "GDPC1.csv"
df = read_data(file_name)
# plot_series(df['gdp'])

# take log difference
gdp_log_diff = np.log(df['gdp']).diff().dropna()

#
# ARIMA model
#

# arima_model = ARIMA(gdp_log_diff, order=(2, 0, 2))
# model_fit = arima_model.fit()
# print(model_fit.summary())

# residuals = DataFrame(model_fit.resid)
# residuals.plot()
# pyplot.show(block=False)

# residuals.plot(kind='kde')
# pyplot.show()
# print(residuals.describe())

# rolling forecast
train_size = int(len(gdp_log_diff)*0.9)
train, test = gdp_log_diff[:train_size].values, gdp_log_diff[train_size:].values
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    model = ARIMA(history, order=(2, 0, 2))  # specified by acf/pacf
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    obs = test[i]
    predictions.append(yhat)
    history.append(obs)
    # print('predicted value: {}, true value: {}'.format(yhat, obs))

# evaluate
MAE = mean_absolute_error(test, predictions)
print('MAE of rolling ARIMA model :{}'.format(MAE))


def plot_result():
    plot1, = pyplot.plot(test, color='blue')
    plot2, = pyplot.plot(predictions, color='red')
    pyplot.legend((plot1, plot2), ('obs', 'predicted'))
    pyplot.title('ARIMA forecasting Predicted value and true value')
    pyplot.show()
    return


