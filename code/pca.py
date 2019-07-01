from data_pipeline import main as processed_data
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def standardize_X_data(X_train, X_test):
    """ Standardizes X_train and X_test data
        (that is called by a function that processes the data) using a StandardScaler """

    # Fit StandardScaler to X_train
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Use the fit StandardScaler model to standardize both X_train and X_test
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    return X_train_std, X_test_std


def optimal_principal_components(X_train_std, X_test_std, var_threshold):
    """ Finds the cumulative sum of variance for the Principal Components, caclulates
        of Principal components that explains the amount of variance that is specified
        with var_threshold and returns DataFrame updated with Principal Component X values """

    # Fit a PCA model with scaled data from scaled X_train data
    pca = PCA().fit(X_train_std)

    # Iterations to find optimal number of principal components
    opt_pc = 0
    for idx, var in enumerate(np.cumsum(pca.explained_variance_ratio_)):
        if var > var_threshold:
            opt_pc = idx+1
            break

    # Refit a PCA model with X_train and the optimal number of principal components
    # Transform both X_train and X_test with the optimal, fit PCA model
    pca_opt = PCA(n_components=opt_pc).fit(X_train_std)
    X_train_pca = pca_opt.fit_transform(X_train_std)
    X_test_pca = pca_opt.fit_transform(X_test_std)

    return X_train_pca, X_test_pca


# def plot_scree(X_scaled):
#     """ Plot a Scree plot to confirm the optimal number of Principal Components
#         from previous calculations"""
#
#     # Fit already standardized X data to a vanilla PCA model
#     pca = PCA()
#     X_pca = pca.fit(X_scaled)
#
#     #Plot scree plot
#     sns.set_style("white")
#     fig, ax = plt.subplots(figsize=(8, 8))
#     sns.lineplot(x=np.arange(len(pca.explained_variance_ratio_)), y=np.cumsum(pca.explained_variance_ratio_))
#     sns.lineplot(x=np.arange(len(pca.explained_variance_ratio_)), y=0.9)
#     ax.set_title('Scree Plot For Principal Components', size=20, weight='bold')
#     ax.set_ylabel('Proportion of Explained Variance', size=18)
#     ax.set_xlabel('Number of Principal Components', size=18)
#     ax.legend(labels=(['Cumulative Sum of Variance', 'Variance Threshold']), loc='lower right', prop={'size': 15})


def main():

    # Get data from previous function
    X_train = processed_data()[0]
    X_test = processed_data()[1]
    y_train = np.array(processed_data()[2])
    y_test = np.array(processed_data()[3])

    y_train = np.array(y_train).reshape(len(y_train))
    y_test = np.array(y_test).reshape(len(y_test))


    # Standardize X_train and X_test
    X_train_std, X_test_std = standardize_X_data(X_train, X_test)

    # Return X_train and X_test with optimal number of Principal Components as features
    X_train_pca, X_test_pca = optimal_principal_components(X_train_std, X_test_std, 0.9)

    return X_train_pca, X_test_pca, y_train, y_test



if __name__ == '__main__':
    main()
