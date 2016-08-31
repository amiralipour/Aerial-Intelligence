# Aerial-Intelligence
Here is my solution for the Aerial Intelligence data science challenge.
I used python with pre-existing libraries such as numpy, sktlearn and Pandas.
Using Pandas the csv files were read and stored in the form of data frames.
To aggregate different sources with similar indexes and features, First I created a new feature called ('ID') which is basically the year, state and county of each entry. Then both entries where aggregated into one single data frame.
Based on the challenge the information on the location (ie state county ..) and date is removed from training.
Graphs 1 and 2 are showing the statistical insight to the raw data. it can be seen that there is not a huge outlier distribution and none of the features have a huge correlation with the 'Yield' feature. Although several multicolinearity can be seen between features them selves. 
Now let’s have a look at the missing values:
1. Missing values: 
..*apparentTemperatureMax      0
..*apparentTemperatureMin      0
cloudCover                  0
dewPoint                    0
humidity                    0
precipIntensity             1
precipIntensityMax          1
precipProbability           1
precipAccumulation          0
precipTypeIsRain            0
precipTypeIsSnow            0
precipTypeIsOther           0
pressure                  605
temperatureMax              0
temperatureMin              0
visibility                 46
windBearing                 0
windSpeed                   0
NDVI                        0
DayInSeason                 0
Yield                       0

It can be seen that the number of missing values versus the total number of entries is small so replacing them with their average value is not going to introduce any bias to our model.
Now lets arbitrarily split our data set into 80% train and 20% test and separately normalize them with respect to max-min so that the zeros are kept zero. 
I tried several regression methods which failed to comply with the test results acceptably. Here are the R^2 values (as a measure for the good of fitness) of different methods applied:
R^2 for linear regression: 0.1034
R^2 for K-nn regression: 0.7800
R^2 for Random Forest regression: 0.8132
R^2 for Extra Trees Random Forest regression: 0.8540

It can be seen that regular linear method yields the lowest accuracy in matching the true values. On the other hand, the Extra Tress Random Forest utilizes bootstrap aggregating or bagging method on each of its subsets or so called trees which are chosen randomly (more detailed explanation can be found elsewhere).
Last graph shows the importance of each feature. I can be seen that the day in the season has the most importance while the mostly zero (precipTypeIsOther) has the zero importance. 
