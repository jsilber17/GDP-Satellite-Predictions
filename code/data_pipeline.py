import pandas as pd
import numpy as np
import glob
import cv2
import re
from scipy.ndimage.measurements import center_of_mass
from skimage import io

def read_in_and_tag_images(filepath, search_string):
    """Read in images from a directory and return a list of image arrays and their labels"""

    # Append images to list as 2d numpy arrays and image names
    path = glob.glob(filepath)
    imgs = []
    names = []
    for img in path:
        n = cv2.imread(img)
        imgs.append(n)
        names.append(img)

    # Create a list of cbsas names and zip the names and images to a list
    result = [re.search(search_string, cbsa).group(1) for cbsa in names]
    return list(zip(result, imgs))


def crop_images(list_of_images):
    """Crop each image to a 38x38 image and create a dictionary with the keys as the cropped
       pictures and their labels as the values"""

    cropped_check = []
    cropped_images = []
    names = []

    # Find the centroid of the image and crop the image to become a 38x38 image
    for idx, img in enumerate(list_of_images):
        names.append(img[0])
        centroid = [int(round(i)) for i in center_of_mass(img[1])[0:2]]
        crop = img[1][centroid[0]-19:centroid[0]+19, centroid[1]-19:centroid[1]+19]
        cropped_images.append(crop.flatten())
        cropped_check.append(crop)

    # Check all shapes of all of the images and return a dictionary with names as keys and cropped images as values
    for img in cropped_check:
        if img.shape != (38, 38, 3):
            raise Exception('One or more cropped images did not have the size (38, 38, 0)')
        else:
            return dict(list(zip(names, cropped_images)))


def create_dataframe(image_dict, y_filepath):
    """Takes the image dictionary and GDP values and merges them into one DataFrame"""

    # Create DataFrame of features
    dfX = pd.DataFrame(data=image_dict).T.reset_index()
    dfX.rename(columns={'index':'city'}, inplace=True)
    col_headers = ['p' + str(col) for col in dfX.columns]
    col_headers.insert(0, 'cbsa')
    col_headers.remove('pcity')
    dfX.columns = col_headers
    dfX = dfX.set_index("cbsa")

    # Average Pixel Intensity
    dfX['average_pixel_intensity'] = [sum(dfX.iloc[cbsa, 0:4332])/len(dfX.columns[0:4332]) for cbsa in range(len(dfX))]

    # Pixel max_pixel_intensity
    dfX['pixel_intensity'] = [sum(dfX.iloc[cbsa, 0:4332]) for cbsa in range(len(dfX))]

    # Max Pixel Intensity
    dfX['max_pixel_intensity'] = [max(dfX.iloc[cbsa, 0:4332]) for cbsa in range(len(dfX))]

    # Create Dataframe of targets
    dfy = pd.read_csv(y_filepath)

    # Creating and dummying GDP sizes
    below, above = np.percentile(dfy['gdp'], [(100/3), (200/3)])
    dfX['gdp_size'] = np.where(dfy['gdp'] < below, 'small', (np.where(dfy['gdp'] > above, 'large', 'medium')))
    dfX['small_gdp'] = np.where(dfX['gdp_size'] == 'small', 1, 0)
    dfX['medium_gdp'] = np.where(dfX['gdp_size'] == 'medium', 1, 0)
    dfX.drop('gdp_size', axis=1, inplace=True)

    # Merge feature and target DataFrame and check for nulls
    result = pd.merge(dfX, dfy, on='cbsa', how='left')
    result = result.set_index(result.columns[0])

    # Order by CBSA and GDP
    result = result.reset_index()
    cbsa = [result['cbsa'][i][:-5] for i in range(len(result))]
    result['cbsa_2'] = cbsa
    df = pd.DataFrame(result.groupby('cbsa_2').sum().sort_values(by='gdp', ascending=False).iloc[: ,-1])
    new = pd.merge(result, df, on='cbsa_2', how='left')
    new = new.sort_values(by='gdp_y', ascending=False).drop(['cbsa_2', 'gdp_y'],axis=1).rename(index=str, columns={"gdp_x": "gdp"}).set_index('cbsa')

    # Check for nulls
    nulls = new[new.isnull().any(axis=1)]
    if len(nulls) == 0:
        return(new)
    else:
        raise Exception("There are nulls in your data.")



def test_train_split(df):

    # Create X and y DataFrames
    X = df.iloc[0:, 0:-1]
    y = pd.DataFrame(df.iloc[0:, -1])

    # Make every three cities train and every fourth city test
    train, test = [], []
    counter = 0
    for i in range(0, len(df)):
        if i == 0:
            counter+=1
            train.append(i)
        elif counter == 9 or counter == 10:
            test.append(i)
            counter+=1
        elif counter == 11:
            test.append(i)
            counter = 0
        else:
            train.append(i)
            counter += 1
    train = np.array(train)
    test = np.array(test)

    # Create Test Train Split
    X_train = X.iloc[train, :]
    X_test = X.iloc[test, :]
    y_train = y.iloc[train, :]
    y_test = y.iloc[test, :]

    return X_train, X_test, y_train, y_test


def main():
    # Define the image path and data variables
    images_filepath = '../satellite_images/*.tif'
    gdp_filepath = '../data/updated_gdp_values.csv'
    search_string = '../satellite_images/(.*).tif'

    # Crop images and create dictionary of images and their respective cbsa names
    satellite_images = read_in_and_tag_images(images_filepath, search_string)
    cropped_images = crop_images(satellite_images)

    # Return a DataFrame that has individual pixels as X columns, cities as rows, and GDP as y column
    df_images = create_dataframe(cropped_images, gdp_filepath)

    X_train, X_test, y_train, y_test = test_train_split(df_images)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    main()

main()





# X_train, X_test, y_train, y_test = test_train_split(df_images)
#
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import Lasso
#
# x = cross_val_score(estimator=Lasso(), X=X_train, y=y_train, scoring='neg_mean_squard_error', cv=10)
# x
#
# from sklearn.model_selection import KFold
#
# from sklearn.metrics import mean_squared_error
#
#
# from sklearn.moel_selection import cross_validate
#
# def crossval(X, y, K, n_estimators):
#     X = np.array(X)
#     y = np.array(y)
#     rsme_list = []
#     x = KFold(n_splits=K)
#     clf = RandomForestRegressor(n_estimators)
#     for train, test in x.split(X):
#         clf.fit(X[train], y[train])
#         test_predicted = clf.predict(X[test])
#         rsme_list.append(np.sqrt(mean_squared_error(test_predicted, y[test])))
#         avg_val = np.array(rsme_list).mean()
#     return avg_val
#
#
# for i in range(100, 1000, 100):
#     avg_val = []
#     avg_val.append(crossval(X_train, y_train, 10, i))
# x = t
#
# for train_index, test_index in kf.split(X_train):
#     # print(len(test_index), len(train_index))
#     # print("TRAIN:", train_index, "TEST:", test_index)
#     X_tr, X_te = np.array(X_train)[train_index], np.array(X_train)[test_index]
#     y_tr, y_te = np.array(y_train)[train_index], np.array(y_train)[test_index]
