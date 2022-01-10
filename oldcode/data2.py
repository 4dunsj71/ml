import yfinance as yf
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import csv
import datetime as dt
import pmdarima as pm
#import logging
#import sys
from sklearn import linear_model
from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression
from pandas.plotting import scatter_matrix
from pmdarima.model_selection import train_test_split
from pandas import read_csv
from matplotlib import pyplot as plt
#from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, confusion_matrix, classification_report, silhouette_score
#from statsmodels.tsa.stattools import adfuller
#from numpy import log
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, normalize
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import NearestNeighbors
#from sklearn.decomposition import PCA

data = read_csv('csv/SnP500Close.csv')
#data = read_csv('SnP500Close.csv', header=None, skiprows=1)



data.fillna(data.mean())

empty = data[data.isna().any(axis=1)]
print('empty shape:\n',empty.shape)

data = data.dropna()

#empty = data[data.isna().any(axis=1)]

#print('empty shape:\n',empty.shape)

#print('empty length:',len(empty))

#print('empty:\n',empty)



#data[0] = pd.to_datetime(data[0])

#data[0] = data[0].astype('int64').astype(float)

#print(data.head(5))


#transpose to make dates the head for the dataframe, to make kmeans cluster for companies not for dates

data = data.T

#new dataframe created to change date string to datetime, then float for easier calculatioons.

dates = data.iloc[0]

dates = pd.to_datetime(dates)

dates = dates.astype('int64').astype(float)

data.iloc[0] = dates

data = data.astype(float)

#x_data = np.array(data.drop(data[0]).astype(float))


#np.array(data.drop([0],1).astype(float))


#x_data = data

#x_data = x_data.drop(data.iloc[0])

#print(x_data)


#np.array(data.drop([0],1).astype(float))

#x_data = x_data.round(decimals = 2)

#print('x_data\n: ',x_data, x_data.shape)

#minmax = MinMaxScaler()

#x_scaled = minmax.fit_transform(x_data)

#print('x_scaled shape:\n',x_scaled.shape)


#y_data = np.array(data[0])

#print('y_data:\n', y_data)

#print('y_data shape:',  y_data.shape)

#----------------------------------------------------------------------------------------------------
#Elbow Bethod
#----------------------------------------------------------------------------------------------------
#data = data.round(5)

#x_data = data.drop(['Date'])
#clust_list = []

#for cluster in range(1,25):

#    kmeans = KMeans(n_clusters = cluster, init='k-means++')

#    kmeans.fit(x_data)

#    clust_list.append(kmeans.inertia_)



#frame =pd.DataFrame({'Cluster':range(1,25), 'cluster_list':clust_list})
#plt.xticks(range(1,25))
#plt.grid(True)
#plt.plot(frame['Cluster'], frame['cluster_list'], marker = 'o')

#plt.xlabel('number of clusters')

#plt.ylabel('Inertia')

#plt.show()

#----------------------------------------------------------------------------------------------------

#Kmeans prediction

#----------------------------------------------------------------------------------------------------

#kmeans = KMeans(n_clusters = 5, init='k-means++')

#kmeans.fit(x_scaled)

#pred = kmeans.predict(x_scaled)

#frame = pd.DataFrame(x_scaled)

#frame['cluster'] = pred

#print(frame['cluster'].value_counts())

#data = data.round(5)

kmeans = KMeans(n_clusters = 5, init='k-means++')
x_data = data.drop(['Date'])
print('x_data\n',x_data)
kmeans.fit(x_data)

pred = kmeans.predict(x_data)

frame = pd.DataFrame(x_data)

frame['cluster'] = pred

print(frame['cluster'].value_counts())

data['cluster'] = frame['cluster']

data['cluster'].astype('category')

for cluster in data['cluster']:
    newFrame = data[data['cluster']== cluster]
    newFrame.to_csv('csv/{}_cluster.csv'.format(cluster))


for  cluster in range(0,4):
    newFrame = read_csv('csv/{}.0_cluster.csv'.format(cluster))
    clustAvg = pd.DataFrame(index = np.arange(1), columns = np.arange(newFrame.shape[1])) 
    col = newFrame.shape[1]
    newFrame = newFrame.iloc[:, 1:col]
    col = newFrame.shape[1]
    
    for ticker in range(0,col):
        avg = newFrame.iloc[:,ticker].mean()
        clustAvg.iloc[:,ticker] = avg
    clustAvg.to_csv('csv/{}_average.csv'.format(cluster))


#----------------------------------------------------------------------------------------------------

#Spectral Clustering

#----------------------------------------------------------------------------------------------------

#x_normal = normalize(x_scaled)

#data_normal = pd.DataFrame(x_normal)

#sc = SpectralClustering(n_clusters=5, affinity='nearest_neighbors',random_state=0)
#sc_clustering = sc.fit(x_normal)


#plt.scatter(x_normal[:0], x_normal[:1], c=sc_clustering.labels_, cmap='rainbow', alpha=0.7, edgecolors='b')
