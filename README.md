## Predicting GDP for CBSA Regions Using Nighttime Satellite Images 

### Question 
NOAA, the National Oceanic and Atmospheric Administration, has been capturing images of the earth since 2012. I stumbled upon their website a few weeks ago and started to browse. I discovered that NOAA has created a webpage with nighttime satellite image tiff files that the public is free to download and use. I started thinking about how these images could be used for machine learning and what the applications of such a project would be. Nigghtime images display lights from different cities and regions across the world... those images are technically 2d arrays with pixel values ranging from 0 to 255 on greyscale... I realized I could use these images to make predictions about the United States! Can satellite images be used to predict the GDP of CBSA regions in the United States? I aim to create multiple machine learning models that attempt to answer that question and choose the best one. 

### Data
#### Google Earth Engine -- Features 
Google Earth Engine offers an API that allows developers to use Google Earth to "detect changes, map trends, and quantify differences on the Earth's surface". I created a Javascript script that interacts with the Earth Engine API and downloads average nighttime images of 133 CBSA regions in the United States for the years 2015-2017, resulting in 399 total tiff images. I saved the images locally, uploaded in a script using glob, cropped them around their centroid into 38x38 pixel images, flattened the 2d image arrays into 1d image arrays, and put them into a Pandas DataFrame with each pixel as a feature. 

<p align="center">
  <img width="800" height="400" src="img/nyc.png">
</p>

#### Bureau of Economic Analysis -- Targets
The Bureau of Economic Analysis has GDP numbers for all CBSA regions in the United States for the years 2011-2019. I downloaded a CSV file of the metrics from the BEA website, uploaded the file to a Pandas DataFrame, and used those values as my targets.

### EDA
It is difficult to do helpful and easily digestible data exploration having pixels as my features, but I found trends that confirmed my belief that there is a relationship between the images and their respective GDPs. 

Bucketing GDP values into small, medium, and large buckets (1/3 of the data for each bucket), finding the average pixel values for those buckets, and plotting them on a histogram shows strong evidence that there is a relational signal in this data. 

![Signal_Histogram](img/Signal_Histogram.png)

### PCA 
Realizing that my features as pixels were hard to interpret, I decided to do Principal Component Analysis on my analysis. PCA uses orthogonal transformation to turn correlated non-linear variables into uncorrelated linear variables and is used for feature reduction in machine learning. I standardized both my train and test features using a StandardScaler and fit them on a PCA model. Once I fit them on a PCA model, I created a Scree chart to find the number of Principal Components that described 90% of the variance in my data. 

![PCA](img/Scree_Plot.png)

The algorithm I used to calculate the number of Principal Components that described 90% of the variance of my data returned 59 Principal Components as features. This can be seen on the Scree plot above. I created a scatterplot of the first two PCs and colored in the data points with their respective GDP sizes (small, medium and large). As seen in the chart below, just the first two PCs begin to show a linear relationship and the GDP sizes are beginning to separate indicating a signal in the data. 

![tpc](img/Two_Principal_Components.png)
