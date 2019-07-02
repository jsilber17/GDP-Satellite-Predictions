from data_pipeline import main as processed_data
from pca import main as pca_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class GDPPredictor(object):
    def __init__(self, model, pca=True):
        self.pca = pca
        if self.pca == True:
            self.X_train = pca_data()[0]
            self.X_test = pca_data()[1]
            self.y_train = pca_data()[2]
            self.y_test = pca_data()[3]
        else:
            self.X_train = processed_data()[0]
            self.X_test = processed_data()[1]
            self.y_train = np.log(processed_data()[2])
            self.y_test = np.log(processed_data()[3])
        self.model = model

    def cross_validate(self, K=10):
        X = np.array(self.X_train)
        y = np.array(self.y_train)
        rsme_list = []
        x = KFold(n_splits=K)
        clf = self.model
        for train, test in x.split(X):
            clf.fit(X[train], y[train])
            test_predicted = clf.predict(X[test])
            rsme_list.append(mean_squared_error(test_predicted, y[test]))
            avg_val = np.array(rsme_list).mean()
        return avg_val


    def changing_Principal_Components(self, num_components):
        if self.pca:
            self.X_train = self.X_train[:, 0:num_components]
            self.X_test = self.X_test[:, 0:num_components]
        else:
            print("You're not using PCA data!")


    def return_train_and_test_errors(self):
        model = self.model.fit(self.X_train, self.y_train)

        y_hat_train = model.predict(self.X_train)
        r2_train = r2_score(np.exp(self.y_train), np.exp(y_hat_train))
        rmse_train = np.sqrt(mean_squared_error(np.exp(self.y_train), np.exp(y_hat_train)))

        y_hat_test = model.predict(self.X_test)
        r2_test = r2_score(np.exp(self.y_test), np.exp(y_hat_test))
        rmse_test = np.sqrt(mean_squared_error(np.exp(self.y_test), np.exp(y_hat_test)))

        return [r2_train, rmse_train, r2_test, rmse_test]



# x = GDPPredictor(RandomForestRegressor(n_estimators=300, max_depth=10, max_features=5))
# x.cross_validate()
# x.changing_Principal_Components(5)
# x.return_train_and_test_errors()
