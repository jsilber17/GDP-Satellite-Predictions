//1.)
//GET IMAGES PER MONTH FOR A SPECIFIC TIMEFRAME
//Change 'region' and 'rectangle' to get images of different geographical regions
//Change 'startDate' and 'finishDate' to get images of different time frames for images
var region = '[[44.3, 15.5], [44.1, 15.5], [44.3, 15.2], [44.1, 15.2]]'
var startDate = '2015-01-01'; //YYYY-MM-DD
var finishDate = '2015-12-31';
var rectangle = ee.Geometry.Rectangle(44.3, 15.5, 44.1, 15.2)


var dataset = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG')
            .filterBounds(rectangle)
            .filterDate(startDate, finishDate)
            .sort('system:time_start', true);
var dataset_med = dataset.median()

//printing the data
var data = dataset.toList(dataset.size());
print(ee.Image(data.get(0)));
var datalen = data.length;
for (var i=0; i<=11; i++){
  var image = ee.Image(data.get(i));
    image = image.select('avg_rad')
      var scaled = image.unitScale(0, 60).multiply(255).toByte();
        Map.addLayer(scaled.clip(rectangle), {min: 0, max: 255}, 'Nighttime');
          print(scaled.clip(rectangle).getDownloadURL({region: region, scale: 30}));
                }
//2.)
//GET AVG IMAGE FOR A SPECIFIED PERIOD OF TIME 
//Change 'region' and 'rectangle' to get images of different geographical regions
//Change 'startDate' and 'finishDate' to get images of different time frames for images
var region = '[[39.09, 36.05], [38.89, 36.05], [39.01, 35.85], [38.89, 35.85]]'
var startDate = '2013-05-01'; //YYYY-MM-DD
var finishDate = '2014-08-31';
var rectangle = ee.Geometry.Rectangle(39.09, 36.05, 38.89, 35.85)

var dataset = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG')
            .filterBounds(rectangle)
            .filterDate(startDate, finishDate)
            .sort('system:time_start', true);
var dataset_mean = ee.Image(dataset.mean())
var image = dataset_mean.select('avg_rad')
var scaled = image.unitScale(0, 60).multiply(255).toByte();
Map.addLayer(scaled.clip(rectangle), {min: 0, max: 255}, 'Nighttime');
print(scaled.clip(rectangle).getDownloadURL({region: region, scale: 30}));

//3.)
//Get the CBSA area and overlay it with Nighttime Images
//Has not worked with MultiPolygons (like Denver)
//Does not use bounding box 
var cityname = 'Rochester, NY';
var city = table.filter(ee.Filter.eq('NAME', cityname));
Map.addLayer(city, {}, 'CBSA');
// var region = '[[-78.8613,44.0703], [-76.3563,44.0703], [-78.8613,42.2429], [-76.3563,42.2429]]';

var polygon = city.geometry()
var region = ee.Geometry(polygon.getInfo()).toGeoJSONString()
var startDate = '2013-05-01'; //YYYY-MM-DD
var finishDate = '2014-08-31';
var dataset = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG')
            .filterBounds(polygon)
            .filterDate(startDate, finishDate)
            .sort('system:time_start', true);
var dataset_mean = ee.Image(dataset.mean())
var image = dataset_mean.select('avg_rad')
var scaled = image.unitScale(0, 60).multiply(255).toByte();
Map.addLayer(scaled.clip(polygon), {min: 0, max: 255}, 'Nighttime');
print(scaled.clip(polygon).getDownloadURL({region:region , scale: 10}));


//4.)
//Get the CBSA area and overlay it with Nighttime Images
//DOES work with MultiPolygons
//Uses a bounding box
var cityname = 'Denver-Aurora-Lakewood, CO';
var city = table.filter(ee.Filter.eq('NAME', cityname));
Map.addLayer(city, {}, 'CBSA');

//Create region for exporting
var polygon = city.geometry()
var boundingBox = polygon.bounds(1)
Map.addLayer(boundingBox)
var region = ee.Geometry(boundingBox.getInfo()).toGeoJSONString()

//Specify Start and End Date for Picture 
var startDate = '2013-05-01'; //YYYY-MM-DD
var finishDate = '2014-08-31';

var dataset = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG')
            .filterBounds(polygon)
            .filterDate(startDate, finishDate)
            .sort('system:time_start', true);
var dataset_mean = ee.Image(dataset.mean())
var image = dataset_mean.select('avg_rad')
var scaled = image.unitScale(0, 60).multiply(255).toByte();
Map.addLayer(scaled.clip(polygon), {min: 0, max: 255}, 'Nighttime');
Map.setCenter(-104.9903, 39.7392, 8)
print(scaled.clip(polygon).getDownloadURL({region:region , scale: 30}));


//5.)
//Get pixel value 
var cityname = 'Denver-Aurora-Lakewood, CO';
var city = table.filter(ee.Filter.eq('NAME', cityname));
Map.addLayer(city, {}, 'CBSA');
// var region = '[[-78.8613,44.0703], [-76.3563,44.0703], [-78.8613,42.2429], [-76.3563,42.2429]]';

var polygon = city.geometry()
var startDate = '2018-01-01'; //YYYY-MM-DD
var finishDate = '2018-12-31';
var dataset = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG')
            .filterBounds(polygon)
            .filterDate(startDate, finishDate)
            .sort('system:time_start', true);
var dataset_mean = ee.Image(dataset.mean())
var image = dataset_mean.select('avg_rad')
var scaled = image.unitScale(0, 60).multiply(255).toByte();
Map.addLayer(scaled.clip(polygon), {min: 0, max: 255}, 'Nighttime');

var stats = scaled.reduceRegion({
  reducer: ee.Reducer.mean(), 
  geometry: polygon, 
  scale: 30, 
  maxPixels: 1e9,
});
print(stats)


//6.) 
//The way I got the images
//Get the CBSA area and overlay it with Nighttime Images
//DOES work with MultiPolygons
//Uses a bounding box
var cityname = 'Anchorage, AK';
var city = table.filter(ee.Filter.eq('NAME', cityname));
Map.addLayer(city, {}, 'CBSA');
print(cityname)
//Create region for exporting
var polygon = city.geometry()
var boundingBox = polygon.bounds(1)
Map.addLayer(boundingBox)
var region = ee.Geometry(boundingBox.getInfo()).toGeoJSONString()

//Specify Start and End Date for Picture 
var startDate = '2015-01-01'; //YYYY-MM-DD
var finishDate = '2015-12-31';
print(startDate)
var dataset = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG')
            .filterBounds(polygon)
            .filterDate(startDate, finishDate)
            .sort('system:time_start', true);
var dataset_mean = ee.Image(dataset.mean())
var image = dataset_mean.select('avg_rad')
var scaled = image.unitScale(0, 60).multiply(255).toByte();
Map.addLayer(scaled.clip(polygon), {min: 0, max: 255}, 'Nighttime');
print(scaled.clip(polygon).getDownloadURL({region:region , scale: 1000}));

var startDate = '2016-01-01'; //YYYY-MM-DD
var finishDate = '2016-12-31';
print(startDate)
var dataset = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG')
            .filterBounds(polygon)
            .filterDate(startDate, finishDate)
            .sort('system:time_start', true);
var dataset_mean = ee.Image(dataset.mean())
var image = dataset_mean.select('avg_rad')
var scaled = image.unitScale(0, 60).multiply(255).toByte();
Map.addLayer(scaled.clip(polygon), {min: 0, max: 255}, 'Nighttime');
print(scaled.clip(polygon).getDownloadURL({region:region , scale: 1000}));

var startDate = '2017-01-01'; //YYYY-MM-DD
var finishDate = '2017-12-31';
print(startDate)
var dataset = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG')
            .filterBounds(polygon)
            .filterDate(startDate, finishDate)
            .sort('system:time_start', true);
var dataset_mean = ee.Image(dataset.mean())
var image = dataset_mean.select('avg_rad')
var scaled = image.unitScale(0, 60).multiply(255).toByte();
Map.addLayer(scaled.clip(polygon), {min: 0, max: 255}, 'Nighttime');
print(scaled.clip(polygon).getDownloadURL({region:region , scale: 1000}));


