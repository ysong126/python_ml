

# data set: Boston Housing

# This script shows how linear model with l2 regulatization can be used in a supervised learning setting. 
# Use Sklearn Cross validation module for hyperparameter tuning

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, r2_score

# cross validation
from sklearn.linear_model import RidgeCV

# load data
# return_X_y option separates feature/target
boston_X_full, boston_Y = datasets.load_boston(return_X_y=True)

# one feature linear regression medv = k + b*crim
# newaxis increases the dimension
# boston_X = boston_X_full[:,np.newaxis,0]

# or, equivalently reshape the data
boston_X = boston_X_full[:, 0]
boston_X = boston_X[:, np.newaxis]

# split into train /test
boston_X_train = boston_X[:-50]
boston_X_test = boston_X[-50:]

# target
boston_Y_train = boston_Y[:-50]
boston_Y_test = boston_Y[-50:]

# object reg stores the model
# fit/test of linear regression takes 2-D arrays for both X and y
# e.g.
# X,y = [[1.5],[2],[2.5]] , [[2],[3],[4]]


reg = linear_model.LinearRegression()
reg.fit(boston_X_train, boston_Y_train)

# predict on test dataset
boston_Y_pred = reg.predict(boston_X_test)

# coef
print('Coef:\n', reg.coef_)

# mse
print('mean squared error: %.2f' % mean_squared_error(boston_Y_test, boston_Y_pred))

# determination
print('coef R squared %.2f' % r2_score(boston_Y_test, boston_Y_pred))

# plot

plt.figure()
plt.scatter(boston_X_test, boston_Y_test,color='black')
plt.plot(boston_X_test, boston_Y_pred, color='blue', linewidth=3)
plt.xlabel('crime rate %')
plt.ylabel('median value of housing prices')
plt.show()


# a ridge regression

# all variables in boston housing data set

boston_X_train_ridge = boston_X_full[:-50, :]
n_alphas = 100
alphas = np.logspace(4,6,n_alphas)

coefs =[]

for a in alphas:
    ridge = linear_model.Ridge(alpha=a)
    ridge.fit(boston_X_train_ridge, boston_Y_train)
    coefs.append(ridge.coef_)

# plot
plt.figure()
ax = plt.gca()
ax.plot(alphas,coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1]) # reverse axis
plt.xlabel('alpha')
plt.ylabel('weight')
plt.title('ridge coefficients as function of the regularization')
plt.axis('tight')
plt.show()

# Cross Validation

'''
a standard workflow is as the following
pick the variables 
run a ridge(or other shrinkage algorithm, lasso/e-net)
run a cross validation to pick the best alpha and the coefficients
plot to visualize
'''

ridge_cv = RidgeCV(alphas = np.logspace(3,6,n_alphas))
ridge_cv.fit(boston_X_train_ridge,boston_Y_train)
print('the best ridge coef is: \n')
print(ridge_cv.coef_ )
print('the best ridge alpha is\n')
print(ridge_cv.alpha_)
