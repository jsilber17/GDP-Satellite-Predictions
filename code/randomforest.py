from data_pipeline import main as processed_data, test_train_split
from pca import main as pca_data
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns



# Functions
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


# Load in Data
X_train, X_test, y_train, y_test = pca_data()


# Use vanilla RandomForest model
rf = RandomForestRegressor(n_estimators=300) # setting trees to a large number

# Cross validate the model
cv_scores = crossval(X_train, y_train, 10, rf) # 0.75

# Trying it out with only 10 PCA features
cv_scores_10 = crossval(X_train[:, 0:10], y_train, 10, rf) #0.73

# Trying it out with only 5 PCA features
cv_scores_5 = crossval(X_train[:, 0:5], y_train, 10, rf) # 0.69

# Using 5 PCA Features

# Hypertuning parameters
param_grid = {'max_depth': [10, 20, 30, 40, 80, 100],
             'max_features': ['auto', 'sqrt', 1, 3, 5]}


# Max Depth
md = [crossval(X_train[:, 0:5], y_train, 10, RandomForestRegressor(n_estimators=300, max_depth=num)) for num in param_grid['max_depth']]
max_depth = param_grid['max_depth'][md.index(np.min(md))]

# Max Features
mf = [crossval(X_train[:, 0:5], y_train, 10, RandomForestRegressor(n_estimators=300, max_features=num)) for num in param_grid['max_features']]
max_features = param_grid['max_features'][mf.index(np.min(mf))]

# Test new model
rf2 = RandomForestRegressor(n_estimators=300, max_depth=max_depth, max_features=max_features)
# Cross Validate new model
rf2_cv = crossval(X_train[:, 0:5], y_train, 10, rf2)
rf2_cv #0.655


# Using 10 PCA Features

# Max Depth
md = [crossval(X_train[:, 0:10], y_train, 10, RandomForestRegressor(n_estimators=300, max_depth=num)) for num in param_grid['max_depth']]
max_depth = param_grid['max_depth'][md.index(np.min(md))]
max_depth
# Max Features
mf = [crossval(X_train[:, 0:10], y_train, 10, RandomForestRegressor(n_estimators=300, max_features=num)) for num in param_grid['max_features']]
max_features = param_grid['max_features'][mf.index(np.min(mf))]
max_features
# Test New Model
rf3 = RandomForestRegressor(n_estimators=300, max_depth=max_depth, max_features=max_features)

rf3_cv = crossval(X_train[:, 0:10], y_train, 10, rf3)
rf3_cv #0.69


# Use 5 PCA Components on Test Data
model = rf2.fit(X_train[:, 0:5], y_train)
y_hat_train = model.predict(X_train[:, 0:5])
r2_score(np.exp(y_train), np.exp(y_hat_train))
np.sqrt(mean_squared_error(np.exp(y_train), np.exp(y_hat_train)))

y_hat_test = rf2.predict(X_test[:, 0:5])
np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_hat_test)))
r2_score(np.exp(y_test), np.exp(y_hat_test))


def main():
    X_train, X_test, y_train, y_test = pca_data()
    rf = RandomForestRegression()
if __name__ == '__main__':
    main()
