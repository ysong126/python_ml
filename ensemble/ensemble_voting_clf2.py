# This script provides an example that uses ensemble (sklearn's voting_classifer)
# Three classifiers used: svm/decision tree/ knn



import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from itertools import product

# load dataset
X, y = datasets.load_iris(return_X_y=True)

np.random.seed(2020)
# split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 1st approach decision tree
tree_classifier = tree.DecisionTreeClassifier(max_depth=4)

# 2nd approach KNN
knn_classifier = neighbors.KNeighborsClassifier(n_neighbors=6)

# 3rd approach SVM
svm_classifier =svm.SVC(gamma=0.1, kernel ='rbf', probability=True)

# training
tree_classifier.fit(X_train, y_train)
knn_classifier.fit(X_train, y_train)
svm_classifier.fit(X_train, y_train)  # first two features only
ensemble_classifier = VotingClassifier(estimators=[('dt', tree_classifier), ('knn', knn_classifier), ('svc', svm_classifier)], voting='soft', weights=[1, 1, 1]).fit(X_train, y_train)

# testing
tree_predict = tree_classifier.predict(X_test)
knn_predict = knn_classifier.predict(X_test)
svm_predict = svm_classifier.predict(X_test)
ensemble_predict = ensemble_classifier.predict(X_test)

# printing out the result, old style string format
print("The accuracy score of decision tree is %.2f" % accuracy_score(tree_predict, y_test))
print("The accuracy score of KNN is %.2f" % accuracy_score(knn_predict, y_test))
print("The accuracy score of SVM is %.2f" % accuracy_score(svm_predict, y_test))
print("The accuracy score of ensemble is %.2f" % accuracy_score(ensemble_predict, y_test))


#
# use entire dataset for training with 2 features with visualization
#

X_full, y = datasets.load_iris(return_X_y=True)
iris = datasets.load_iris()
X = X_full[:, :2]
feature_names =iris.feature_names[:2]
classes = iris.target_names

def make_meshgrid(x, y, h =.02):
    x_min, x_max = x.min()-1, x.max()+1
    y_min, y_max = y.min()-1, y.max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

fig, ax = plt.subplots(2,2, sharex= 'col', sharey='row', figsize=(12, 9))
# first two features
xx, yy = make_meshgrid(X[:,0], X[:, 1])

# in-sample training with 2 features only for visualization
tree_classifier.fit(X[:, :2], y)
knn_classifier.fit(X[:, :2], y)
svm_classifier.fit(X[:, :2], y)  # first two features only
ensemble_classifier.fit(X[:, :2], y)

# plotting the boundaries
for idx, clf, t in zip(product([0, 1], [0, 1]), [tree_classifier, knn_classifier, svm_classifier, ensemble_classifier], ["Tree", "KNN", "SVM", "Soft Voting"]):
    # split region
    plot_contours(ax[idx[0], idx[1]], clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.5)
    # plot the data points
    ax[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    ax[idx[0], idx[1]].set_title(t)
    ax[idx[0], idx[1]].set_ylabel("{}".format(feature_names[1]))
    ax[idx[0], idx[1]].set_ylabel("{}".format(feature_names[0]))

plt.show()


