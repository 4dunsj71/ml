import yfinance as yf
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import csv
import datetime as dt
import pmdarima as pm
from sklearn import linear_model
from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression
from pandas.plotting import scatter_matrix
from pmdarima.model_selection import train_test_split
from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, confusion_matrix, classification_report, silhouette_score
from statsmodels.tsa.stattools import adfuller
from numpy import log
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, normalize
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


data = read_csv('SnP500Close.csv', header=None, skiprows=1)

empty = data[data.isna().any(axis=1)]
print(empty.shape)
print(len(empty))
print(empty)
data.fillna(data.mean(),inplace=True)


data[0] = pd.to_datetime(data[0])
data[0] = data[0].astype('int64').astype(float)
x_data = np.array(data.drop([0],1).astype(float))
x_data = x_data.round(decimals = 2)
print(x_data, x_data.shape)
minmax = MinMaxScaler()
x_scaled = minmax.fit_transform(x_data)
print(x_scaled.shape)

y_data = np.array(data[0])
print(y_data.shape)
#----------------------------------------------------------------------------------------------------
#Elbow Bethod
#----------------------------------------------------------------------------------------------------
#clust_list = []

#for cluster in range(1,50):
#    kmeans = KMeans(n_clusters = cluster, init='k-means++')
#    kmeans.fit(x_scaled)
#    clust_list.append(kmeans.inertia_)

#frame =pd.DataFrame({'Cluster':range(1,50), 'cluster_list':clust_list})
#plt.plot(frame['Cluster'], frame['cluster_list'], marker = 'o')
#plt.xlabel('number of clusters')
#plt.ylabel('Inertia')
#plt.show()


#----------------------------------------------------------------------------------------------------
#Kmeans prediction
#----------------------------------------------------------------------------------------------------

kmeans = KMeans(n_clusters = 5, init='k-means++')
kmeans.fit(x_scaled)
pred = kmeans.predict(x_scaled)
frame = pd.DataFrame(x_scaled)
frame['cluster'] = pred
print(frame['cluster'].value_counts())
#----------------------------------------------------------------------------------------------------
#Spectral Clustering
#----------------------------------------------------------------------------------------------------
#x_normal = normalize(x_scaled)
#data_normal = pd.DataFrame(x_normal)

#sc = SpectralClustering(n_clusters=5, affinity='nearest_neighbors',random_state=0)
#sc_clustering = sc.fit(x_normal)

#plt.scatter(x_normal[:0], x_normal[:1], c=sc_clustering.labels_, cmap='rainbow', alpha=0.7, edgecolors='b')
