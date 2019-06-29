from data_pipeline import main as processed_data
from pca import main as pca_data
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
X_train, X_test, y_train, y_test = pca_data()
y_train = np.array(y_train).reshape(len(y_train))
y_test = np.array(y_test).reshape(len(y_test))

rf = RandomForestRegressor(bootstrap=True, criterion='mae', max_depth=5, n_estimators=1000, min_samples_leaf=15, n_jobs=-1)
rf.fit(X_train, y_train)
rf.score(X_train, y_train)
predictions = rf.predict(np.array(X_test))

mean_squared_log_error(predictions, y_test)
r2_score(y_test, predictions)
