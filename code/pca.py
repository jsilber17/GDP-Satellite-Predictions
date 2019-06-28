from data_pipeline import main as processed_data
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def scale_X_data(processed_data_function):
    """ Standardizes X data (that is called by a function that processes the data)
        using a StandardScaler """

    # Isolate X data from processed data
    df = processed_data_function
    index = pd.DataFrame(df.iloc[:, 0])
    df = df.set_index(df.columns[0])
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]

    # Standardize data with StandardScaler --> subtract the mean and divide by standard deviation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, index


def optimal_principal_components(X_scaled, y, index, var_threshold):
    """ Finds the cumulative sum of variance for the Principal Components, caclulates
        of Principal components that explains the amount of variance that is specified
        with var_threshold and returns DataFrame updated with Principal Component X values """

    # Fit a PCA model with scaled data from scale_X_data
    pca = PCA().fit(X_scaled)

    # Iterations to find optimal number of principal components
    opt_pc = 0
    for idx, var in enumerate(np.cumsum(pca.explained_variance_ratio_)):
        if var > var_threshold:
            opt_pc = idx+1
            break

    # Refit a PCA model with the optimal number of principal components
    pca_opt = PCA(n_components=opt_pc)
    X_pca = pd.DataFrame(pca_opt.fit_transform(X_scaled))

    # Create DataFrame with Principal Component X values and columns and the same y values
    pc_df = index.join(X_pca).set_index('cbsa').join(y)

    return pc_df


def plot_scree(X_scaled):
    """ Plot a Scree plot to confirm the optimal number of Principal Components
        from previous calculations"""

    # Fit already standardized X data to a vanilla PCA model
    pca = PCA()
    X_pca = pca.fit(X_scaled)

    #Plot scree plot
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(x=np.arange(len(pca.explained_variance_ratio_)), y=np.cumsum(pca.explained_variance_ratio_))
    sns.lineplot(x=np.arange(len(pca.explained_variance_ratio_)), y=0.9)
    ax.set_title('Scree Plot For Principal Components', size=20, weight='bold')
    ax.set_ylabel('Proportion of Explained Variance', size=18)
    ax.set_xlabel('Number of Principal Components', size=18)
    ax.legend(labels=(['Cumulative Sum of Variance', 'Variance Threshold']), loc='lower right', prop={'size': 15})
    plt.show()


def main():
    # Get pre-optimal X values, y and index
    X_scaled, y, index = scale_X_data(processed_data())

    # Return DataFrame with optimal number of Principal Components as columns
    pc_df = optimal_principal_components(X_scaled, y, index, 0.9)

    #Plot Scree plot to confirm optimal number of Principal Components
    plot_scree(X_scaled)

    return pc_df



if __name__ == '__main__':
    main()
