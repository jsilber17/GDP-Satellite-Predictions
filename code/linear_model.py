from data_pipeline import main as processed_data
from pca import main as pca_data
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score



# def image_feature_engineering(df_pca, df_processed, y):
#     """ Create new features for image data """
#
#     # Load in pre-PCA DataFrame for feature engineering
#     df_engineer = df_processed.iloc[:, 0:-1]
#     df_engineer = df_engineer.set_index(df_engineer.columns[0])
#
#     # Average Pixel Intensity
#     df_pca['average_pixel_intensity'] = [sum(df_engineer.iloc[cbsa, 0:4332])/len(df_engineer.columns[0:4332]) for cbsa in range(len(df_engineer))]
#
#     # Max Pixel Intensity
#     df_pca['max_pixel_intensity'] = [max(df_engineer.iloc[cbsa, 0:4332]) for cbsa in range(len(df_engineer))]
#
#     # Creating and dummying GDP sizes
#     below, above = np.percentile(y, [(100/3), (200/3)])
#     df_pca['gdp_size'] = np.where(y < below, 'small', (np.where(y > above, 'large', 'medium')))
#     df_pca['small_gdp'] = np.where(df_pca['gdp_size'] == 'small', 1, 0)
#     df_pca['medium_gdp'] = np.where(df_pca['gdp_size'] == 'medium', 1, 0)
#     df_pca.drop('gdp_size', axis=1, inplace=True)
#
#     #Standardize X Data for Regularization
#     scaler = StandardScaler()
#     df_pca_scal = pd.DataFrame(scaler.fit_transform(df_pca)).set_index(df_engineer.index)
#     result = pd.merge(df_pca_scal, pd.DataFrame(y), on='cbsa', how='left')
#
#     # Return DataFrame with engineered features
#     return result


def LassoRegression(X_train, y_train, cv):

    # Cross validate to find the optimal alpha
    # cv_scores = [cross_val_score(ElasticNet(alpha), X_train, y_train, scoring='r2', cv=cv).mean() for alpha in np.arange(0, 100, 10)]
    cv_scores = cross_val_score(Lasso(6000), X_train, y_train, scoring='r2', cv=5)


    return cv_scores



def main():
    X_train, X_test, y_train, y_test = pca_data()
    cv_scores = LassoRegression(X_train, y_train, 5)

    print(cv_scores)


if __name__ == '__main__':
    main()
