import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
import patsy
import seaborn as sns
import sklearn.preprocessing as prep
import sklearn.ensemble.forest as sklf
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor

# Setting Display parameters
pd.set_option('display.max_columns', 100)
sns.set(style="white")

# read in the data
df_13 = pd.read_csv("/Users/amiralipourskandani/Downloads/wheat-2013-supervised.csv")
df_14 = pd.read_csv("/Users/amiralipourskandani/Downloads/wheat-2014-supervised.csv")

# To avoid duplicated county names in different states and also make it easier later for aggregating the data,
# an ID index is added to the data explaining which year and what county and state it blongs to
df_13['ID'] = df_13['CountyName'] + ',' + df_13['State'] + ', 2013'
df_14['ID'] = df_14['CountyName'] + ',' + df_14['State'] + ', 2014'
df = df_13.append(df_14)

# Also lets get ride of the location information
df.drop(df.columns[[0,1,2,3,4, 26]], axis=1,inplace=True)


# looking to see if there is any apparent outlier
plt.figure(1)
df.boxplot(['apparentTemperatureMax', 'apparentTemperatureMin', 'cloudCover', 'dewPoint','humidity', 'pressure',
            'temperatureMax', 'temperatureMin', 'windBearing', 'windSpeed', 'NDVI'], return_type='axes')
plt.xticks(rotation=-65)

# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
map = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

plt.yticks(rotation=0)
plt.xticks(rotation=90)

# Missing Values for each column
print('Missing values:',df.apply(lambda x: sum(x.isnull()),axis=0))

# The number of missing values are small compared to the total data so lets replace them with average value
#df_clean = df[np.isfinite(df['pressure'])].reset_index()
imp_1 = prep.Imputer(missing_values='NaN', strategy='mean', axis=0)
df_clean = imp_1.fit_transform(df)


# Splitting the data to train and test
train, test = train_test_split(df_clean, test_size=0.2)


## Predictive Modeling:

# normalizing the data
X_train = train#.values #returns a numpy array
min_max_scaler = prep.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X_train)
train_n = pd.DataFrame(x_scaled)
train_n.columns = df.columns


X_test = test#.values #returns a numpy array
xy_scaled = min_max_scaler.transform(X_test)
test_n = pd.DataFrame(xy_scaled)
test_n.columns = df.columns


# Formulating the model
formu = "Yield ~ apparentTemperatureMax + apparentTemperatureMin + cloudCover + dewPoint + humidity + precipIntensity"\
        "+ precipIntensityMax +	precipProbability +	precipAccumulation + precipTypeIsRain +	precipTypeIsSnow + " \
        "precipTypeIsOther + pressure +	temperatureMax + temperatureMin + visibility + windBearing + windSpeed + NDVI "\
        "+ DayInSeason"





y,X = patsy.dmatrices(formu, train_n, return_type='dataframe')
y1,X1 = patsy.dmatrices(formu, test_n, return_type='dataframe')


#linear model
clf_1 = linear_model.LinearRegression()
clf_1.fit(X,y)
print("R^2 for linear regression:", clf_1.score(X1, y1))

#K neighbor regressor
clf_2 = KNeighborsRegressor()
clf_2.fit(X,y.values.ravel())
print("R^2 for K-nn regression:", clf_2.score(X1, y1.values.ravel()))

#random forerst regressor
clf_3 = sklf.RandomForestRegressor(n_estimators=15)
clf_3.fit_transform(X,y.values.ravel())
print("R^2 for Random Forest regression:", clf_3.score(X1, y1.values.ravel()))

#Extra trees random forerst regressor
clf_4 = sklf.ExtraTreesRegressor(n_estimators=15)
clf_4.fit_transform(X,y.values.ravel())
print("R^2 for Extra Trees Random Forest regression:", clf_4.score(X1, y1.values.ravel()))

# Plot the feature importances of the forest
importances = clf_4.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_4.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), train_n.columns[indices-1])
plt.xlim([-1, X.shape[1]])
plt.xticks(rotation=90)

plt.show()

