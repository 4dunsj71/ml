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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, confusion_matrix, classification_report

tickers=[
        'aapl',
        'msft'
    ]

#pulls historical close prices for tickers listed
#for ticker in tickers:
#    ticker_a = yf.Ticker(ticker)
#    data = ticker_a.history(period='300d', interval='1d')
#    data['Close'].to_csv("close{}.csv".format(ticker))

#read the csv, from file
#obviously temp code, needs to be automated from the ground up to pull and train on ticker user selects

#reading in csv, skipping header row, adding own header to target for model
#data = read_csv("closeaapl.csv",skiprows=[0])
#data_head = ['date','close']

#im just gonna fucking keep the original head what the fuck even was that error

data = read_csv("closeaapl.csv")
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].astype('int64').astype(float)
data['Close'] = data['Close'].astype(float)

#print("date:\n",data['Date'].tail(10),"\nclose:\n",data['Close'].tail(10))



train_head = ['Date']
target_head = ['Close']


X = data[train_head]
y = data[target_head]
#print(X,"\n", y)

x = data['Date'].values.reshape(-1, 1)
y = data['Close'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


model = auto_arima(y_train, m=12, stepwise=True, trace=True, seasonal = True)

model.fit(y_train)

y_predict = model.predict(start=0, n_periods=len(y_train))
print(y_predict)
plt.plot(x_train,y_train)
plt.plot(x_train, y_predict)
plt.show()




