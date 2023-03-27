from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(2020)

# load data and split into training/test
X, y = datasets.load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# standardization
scaler = StandardScaler()
# fit the scaler
scaler.fit(X_train)
# use scaler to transform data
scaled_X_train = scaler.transform(X_train)

# use MLP to classify
neural_net_clf = MLPClassifier()
neural_net_clf.fit(scaled_X_train, y_train)

# standardize test set
scaler2 = StandardScaler()
scaler2.fit(X_test)
scaled_X_test = scaler2.transform(X_test)

# test accuracy
predicted_y = neural_net_clf.predict(scaled_X_test)
acc_score = accuracy_score(predicted_y, y_test)

print("the accuracy score is {}".format(acc_score))