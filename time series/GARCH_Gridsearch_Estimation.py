import numpy as np
import pandas as pd
from pandas import read_csv, DataFrame
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, quarter_plot
from statsmodels.api import qqplot
from scipy.stats import probplot
from arch import arch_model

# data structure   2264 by 2 / DataFrame
# Date(index)   Close
#    date       value
#    date       value


def read_file(file_name):
    df_full = read_csv(file_name, sep=',')
    close = df_full[['Date', 'Close']]
    # take log difference to get return (remove 1st row)
    returns = np.log((close['Close'])).diff().iloc[1:]
    # if taken log then times 100
    returns = 100*returns
    index_date = close['Date'].iloc[1:]
    df = pd.concat([index_date, returns], axis=1).dropna()
    df.set_index(keys=index_date, inplace=True)
    return df


def check_stationary(df, col_name, lags):
    """
    :param df: dataframe nx2 [date, close]
    :param col_name: column name
    :param lags: draw lags in plot
    :return: None
    """
    series = df[col_name]
    fig = pyplot.figure(figsize=(10, 8))
    layout = (3, 2)
    ts_ax = pyplot.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = pyplot.subplot2grid(layout, (1, 0))
    pacf_ax = pyplot.subplot2grid(layout, (1, 1))
    qq_ax = pyplot.subplot2grid(layout, (2, 0))
    pp_ax = pyplot.subplot2grid(layout, (2, 1))

    series.plot(ax=ts_ax)
    ts_ax.set_title("Time Series Analysis Plots")
    plot_acf(series, lags=lags, ax=acf_ax, alpha=0.5)
    plot_pacf(series, lags=lags, ax=pacf_ax, alpha=0.5)
    qqplot(series, line='s', ax=qq_ax)
    probplot(series, sparams=(series.mean(), series.std()), plot=pp_ax)
    pyplot.tight_layout()
    return


def split(df, n_hozizon):
    train_set, test_set = df.iloc[:-n_horizon], df.iloc[-n_horizon:]
    return train_set, test_set


def grid_search(p, q, h):
    """
    grid search on p and q, by AIC criteria

    :param p: order of lag - innovation
    :param q: order of lag - volatility
    :param h: forecast horizon
    :return: [p, q, AIC]
    """
    aic_min = np.inf
    result =[1, 1, aic_min]
    for i in range(1, p):
        for j in range(1, q):
            garch_model_candidate = arch_model(train['Close'].iloc[:-n_horizon], mean="Constant", vol='GARCH', p=i, q=j)
            garch_model_candidate_fit=garch_model_candidate.fit()
            if garch_model_candidate_fit.aic<aic_min:
                aic_min=garch_model_candidate_fit.aic
                result=[i, j, aic_min]
    return result


# read file and check stationarity
close_df = read_file("SP500.csv")
check_stationary(close_df, 'Close', 25)

# split train and forecast
return_mean = np.mean(close_df['Close'])

n_horizon = 10
train, test = split(close_df, n_horizon)

# models
# ARCH
# my_arch_model = arch_model(train['Close'].iloc[:n_train], vol='ARCH', horizon=n_horizon).fit()


# GARCH
# grid search gives (2,2) by AIC criteria
my_garch_model = arch_model(train['Close'].iloc[:-n_horizon], mean="Constant", vol='GARCH', p=2, q=2).fit()

# Other variants
# GJR GARCH:
# my_garch_model = arch_model(train['Close'].iloc[:-n_horizon], mean="Constant", vol='GARCH', p=2, o=1, q=2).fit()

# TARCH
# my_garch_model = arch_model(train['Close'].iloc[:-n_horizon], mean="Constant", p=2, o=1, q=2, power=1.0).fit()

# Student's T
# my_garch_model = arch_model(train['Close'].iloc[:-n_horizon], mean="Constant", p=2, q=2, power=1.0, dist='StudentsT').fit()

garch_yhat = my_garch_model.forecast(horizon=n_horizon)
print(my_garch_model.summary())

# plot
y_var = test['Close']**2
fig = pyplot.figure()
plot_t, = pyplot.plot(y_var)
plot_p, = pyplot.plot(garch_yhat.variance.values[-1, :])
pyplot.legend((plot_t, plot_p), ('Obs {}\u00b2'.format('r'), 'GARCH predicted E[{}\u00b2]'.format('r')),
              loc='center right')
pyplot.xticks(rotation=45)
pyplot.ylim((0, 5))
pyplot.show(block=False)


