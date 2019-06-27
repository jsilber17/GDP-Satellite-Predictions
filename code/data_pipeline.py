import pandas as pd
import numpy as np
import glob
import cv2
import re
from scipy.ndimage.measurements import center_of_mass


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

    # Create Dataframe of targets
    dfy = pd.read_csv(y_filepath)

    # Merge feature and target DataFrame and check for nulls
    result = pd.merge(dfX, dfy, on='cbsa', how='left')
    nulls = result[result.isnull().any(axis=1)]
    if len(nulls) == 0:
        return(result)
    else:
        raise Exception("There are nulls in your data.")


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
    return df_images

if __name__ == '__main__':
    main()
