
#  required functions are imported from sklearn
import pandas as pd
import numpy as np
import matplotlib as plt

# Loading the Boston dataset
df = pd.read_csv('kc_house_data.csv')

### Exploratory Data Analysis
#droping some features
df.info()
df.head()
df.columns
#df = df.drop(['id', 'date', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long'], axis=1)
#Checking for missing Data
df.isnull().sum()

#describe the statistics for each row of the  DataFrame
df.describe().transpose()

# some visualizations using sns
import seaborn as sb # statistical data visualization library based on matplotlib
sb.countplot(x=df['bedrooms'])
sb.scatterplot(x=df['sqft_living'],y= df['price'])
sb.heatmap(df.corr(), cmap="YlGnBu", annot=True)
df = df.drop(['id', 'date', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long','sqft_above'], axis=1)
# Preparing input and output data
X = df.drop('price', axis =1).values #drop output varibale
y = df.price.values

## Train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

## scaling the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
## Scaling the data

X_train_stdsc = scaler.fit_transform(X_train)
X_test_stdsc = scaler.fit_transform(X_test)
from sklearn.neural_network import MLPRegressor
#mlp = MLPRegressor(hidden_layer_sizes=(6,),max_iter=1000)
mlp_reg = MLPRegressor(hidden_layer_sizes=(150,50,20),max_iter = 300,activation = 'relu',solver = 'adam',learning_rate='invscaling',momentum=0.4,verbose=True)
mlp_reg.fit(X_train_stdsc,y_train)
y_pred = mlp_reg.predict(X_test_stdsc)
df_temp = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_temp.head()
