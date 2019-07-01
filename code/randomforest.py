from data_pipeline import main as processed_data
from pca import main as pca_data
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn import metrics
import seaborn as sns


def RandomForestRegression(X_train, y_train):

    # Cross validate to find the optimal alpha
    cv_scores = cross_val_score(RandomForestRegressor(), X_train, np.log(y_train), scoring='r2', cv=5)


    return cv_scores


def main():
    X_train, X_test, y_train, y_test = pca_data()
    cv_scores = RandomForestRegression(X_train, y_train)

    print (cv_scores)

if __name__ == '__main__':
    main()


X_train, X_test, y_train, y_test = pca_data()

y_train = np.log(np.array(y_train).reshape(len(y_train)))
y_test = np.log(np.array(y_test).reshape(len(y_test)))

rf = RandomForestRegressor(bootstrap=True, criterion='mae', max_depth=5, n_estimators=150, min_samples_leaf=15, n_jobs=-1)
rf.fit(X_train, y_train)
rf.score(X_train, y_train)
predictions = rf.predict(np.array(X_test))

metrics.mean_squared_log_error(predictions, y_test)
metrics.r2_score(y_test, predictions)
y_test, predictions
np.sqrt(metrics.mean_squared_error(y_test, predictions))
# param_grid = {'bootstrap': [True, False],
#               'criterion': ['mae', 'mse'],
#               'max_depth': [1, 50, 100],
#               'max_features': ['auto', 'sqrt', 'log2', None],
#               'max_leaf_nodes': [2, 50, 100],
#               'min_samples_split': [2, 50, 100],
#               'n_estimators': [50, 100, 1000]}
# clf = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, n_jobs=-1, return_train_score=True)
# clf.fit(X_train, y_train)
# clf.best_params_
# rf2 = RandomForestRegressor(bootstrap=True, criterion='mae', max_depth=50, max_features='auto', max_leaf_nodes=100, min_samples_split=500, n_estimators=50)
# rf2.fit(X_train, y_train)
# rf2.score(X_train, y_train)

gb = GradientBoostingRegressor(alpha=0.9, learning_rate=0.01)
gb.fit(X_train, y_train)
gb.score(X_train, y_train)
predictions = gb.predict(np.array(X_test))
metrics.mean_squared_log_error(predictions, y_test)
np.sqrt(metrics.mean_squared_error(y_test, predictions))
metrics.r2_score(y_test, predictions)

sns.scatterplot(x=np.arange(len(y_test)), y=np.log(np.array(y_test).reshape(99)))
sns.scatterplot(x=np.arange(len(y_train)), y=np.log(np.array(y_train).reshape(300)))
y_train
