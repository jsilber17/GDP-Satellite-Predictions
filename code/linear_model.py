from data_pipeline import main as processed_data
from pca import main as pca_data
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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


# Load in Data
X_train, X_test, y_train, y_test = pca_data()


# Use vanilla Lasso model
lasso = Lasso()

# Cross validate the model
cv_scores = crossval(X_train, y_train, 10, lasso) # 0.48

# Trying it out with 10 PCA features
cv_scores_10 = crossval(X_train[:, 0:10], y_train, 10, lasso) # 0.48

# Trying it out with 5 PCA features
cv_scores_5 = crossval(X_train[:, 0:5], y_train, 10, lasso) # 0.47

# Cross validating for Alpha with 5 features
alpha = [crossval(X_train[:, 0:5], y_train, 10, Lasso(num)) for num in [0.5, 0.6, 0.7, 0.8, 0.9, 1]]
alpha_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
a = alpha_list[alpha.index(np.min(alpha))] # alpha = 0.8

# Test new model
lasso2 = Lasso(alpha=a)
lasso2_cv = crossval(X_train[:, 0:5], y_train, 10, Lasso(alpha=a)) # 0.47

# Test on Train set
model = lasso2.fit(X_train[:, 0:5], y_train)
y_hat_train = lasso2.predict(X_train[:, 0:5])
r2_score(np.exp(y_train), np.exp(y_hat_train))
np.sqrt(mean_squared_error(np.exp(y_train), np.exp(y_hat_train)))

# Test on Test set
y_hat_test = lasso2.predict(X_test[:, 0:5])
np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_hat_test)))
r2_score(np.exp(y_test), np.exp(y_hat_test))


# Non Feature
X_train, X_test, y_train, y_test = processed_data()
X_train = X_train.iloc[:, 4332:]
X_test = X_test.iloc[:, 4332:]

# Defining the model
lr = Lasso()

#Cross Validation
cv_scores = crossval(X_train, np.log(y_train), 10, lr)
cv_scores

crossval(X_train, np.log(y_train), 10, lr)


model = lr.fit(X_train, np.log(y_train))
y_hat_train = lr.predict(X_train)
np.sqrt(mean_squared_error(y_train, np.exp(y_hat_train)))
r2_score(y_train, np.exp(y_hat_train))

y_hat_test = lr.predict(X_test)
np.sqrt(mean_squared_error(y_test, np.exp(y_hat_test)))
r2_score(y_test, np.exp(y_hat_test))

# def main():
#     X_train, X_test, y_train, y_test = pca_data()
#     X_train_train, X_train_test, y_train_train, y_train_test = second_test_train_split(X_train, y_train)
#     alpha = find_optimal_alpha(X_train_train, y_train_train, X_train_test, y_train_test)
#     y_hat = fit_and_predict(X_train_train, y_train_train, X_train_test, alpha)
#     mae = np.sqrt(mean_squared_error(np.exp(y_train_test), np.exp(y_hat)))
#     r2 = r2_score(np.exp(y_train_test), np.exp(y_hat))
#
#     y_hat2 = fit_and_predict(X_train[:, 0:5], y_train, X_test[:, 0:5], alpha)
#     mae2 = np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_hat2)))
#     r22 = r2_score(np.exp(y_test), np.exp(y_hat2))
#
#     print(mae2, r22)
#
# if __name__ == '__main__':
#     main()
