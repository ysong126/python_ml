# This script provides:
# A LASSO (Linear Absolute Shrinkage Selection Operator ) regression

# 1) Lasso with a pred-determined alpha
# 2) Lasso with cross validation using LARS (select the best alpha/ hyperparameter tuning)
# 3) Visualization - Plot Lasso coefficients as function of the regularization(alpha)

import time

import numpy as np
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

# Boston housing dataset 506 rows, 14 columns
boston_X, boston_Y = datasets.load_boston(return_X_y=True)

# number of obs in test set, the last ____ rows
test_num = 106

# split training set and test set
boston_X_train = boston_X[:-test_num, :]
boston_Y_train = boston_Y[:-test_num]

boston_X_test = boston_X[-test_num:, :]
boston_Y_test = boston_Y[-test_num:]

# lasso regression   alpha determined
lasso_reg = linear_model.Lasso(alpha=0.1, normalize=True)
lasso_reg.fit(boston_X_train, boston_Y_train)

boston_Y_pred = lasso_reg.predict(boston_X_test)

mse = mean_squared_error(boston_Y_pred, boston_Y_test)

# Mean Square Error
print('the mse of the LASSO regression is %f' % mse)


# cross validation using LARS
# NOTE：It's necessary to normalize features for LASSO/Ridge
# linear_model does normalization by default

# timing the training process
start_time=time.time()
lasso_lars_cv = linear_model.LassoLarsCV(cv=10, normalize=True).fit(boston_X_train, boston_Y_train)
end_time=time.time()
training_time = end_time-start_time
boston_Y_pred_cv = lasso_lars_cv.predict(boston_X_test)

mse_cv = mean_squared_error(boston_Y_pred_cv, boston_Y_test)

print('the mse of the LASSO Cv regression is %f, training time %f' % (mse_cv , training_time))

alpha_aic = lasso_lars_cv.alpha_
# plot
# print alphas to see the range  (1e-2,3e-1)
print(lasso_lars_cv.alphas_)

# alpha candidates from [1e-2,1e+2]
alphas = np.logspace(-3, 0, 100)
coefs = []
for a in alphas:
    lasso_reg = linear_model.Lasso(alpha=a, normalize=True)
    lasso_reg.fit(boston_X_train,boston_Y_train)
    coefs.append(lasso_reg.coef_)

# plot
plt.figure()
ax =plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
# reverse the axis- the larger the penalty the fewer coefficients are left
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel(r'$\alpha$')
plt.ylabel('weights/coefficients')
plt.title('Coefficients as a function of alpha (LASSO CV using LARS)')
