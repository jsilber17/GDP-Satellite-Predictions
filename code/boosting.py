from data_pipeline import main as processed_data
from pca import main as pca_data
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns


def crossval(X, y, K, model):
    X = np.array(X)
    y = np.array(y)
    rsme_list = []
    x = KFold(n_splits=K)
    clf = model
    for train, test in x.split(X):
        clf.fit(X[train], y[train])
        test_predicted = clf.predict(X[test])
        rsme_list.append(mean_squared_error(test_predicted, y[test]))
        avg_val = np.array(rsme_list).mean()
    return avg_val


X_train, X_test, y_train, y_test = pca_data()

# Use vanilla Lasso model
gb = GradientBoostingRegressor()

# Cross validate the model
cv_scores = crossval(X_train, y_train, 10, gb) # 0.78

# Trying it out with 10 PCA features
cv_scores_10 = crossval(X_train[:, 0:10], y_train, 10, lasso) # 0.69

# Trying it out with 5 PCA features
cv_scores_4 = crossval(X_train[:, 0:4], y_train, 10, lasso) # 0.64

# Loss
loss = [crossval(X_train[:, 0:4], y_train, 10, GradientBoostingRegressor(loss=num)) for num in ['ls', 'lad', 'huber', 'quantile']]
loss_list = ['ls', 'lad', 'huber', 'quantile']
l = loss_list[loss.index(np.min(loss))]

# Estimators
est = [crossval(X_train[:, 0:4], y_train, 10, GradientBoostingRegressor(loss=l, n_estimators=num)) for num in range(10, 1000, 100)]
estimators = [i for i in range(10, 1000, 100)]
ne = estimators[loss.index(np.min(loss))]

# Learning Rate
learning_rate = [crossval(X_train[:, 0:4], y_train, 10, GradientBoostingRegressor(loss=l, n_estimators=ne, learning_rate=num)) for num in np.arange(0.05, 1, 0.05)]
lr_list = list(np.arange(0.5, 1, 0.05))
lr = lr_list[lr_list.index(np.min(lr_list))]

# Cross val on new model
gb2 = GradientBoostingRegressor(loss=l, n_estimators=ne, learning_rate=lr)
gbcv = crossval(X_train[:, 0:4], y_train, 10, gb2) # 0.47


# Test on Train
model = gb2.fit(X_train[:, 0:4], y_train)
y_hat_train = gb2.predict(X_train[:, 0:4])
r2_score(np.exp(y_train), np.exp(y_hat_train))
np.sqrt(mean_squared_error(np.exp(y_train), np.exp(y_hat_train)))

# Test on Test
y_hat_test = gb2.predict(X_test[:, 0:4])
np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_hat_test)))
r2_score(np.exp(y_test), np.exp(y_hat_test))
