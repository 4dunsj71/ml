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
from statsmodels.tsa.stattools import adfuller
from numpy import log
tickers=[
        'aapl',
        'msft'
    ]

#pulls historical close prices for tickers listed
#for ticker in tickers:
#    ticker_a = yf.Ticker(ticker)
#    data = ticker_a.history(period='300d', interval='1d')
#    data['Close'].to_csv("csv/close{}.csv".format(ticker))

#read the csv, from file
#obviously temp code, needs to be automated from the ground up to pull and train on ticker user selects

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

result = adfuller(y_train)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

d = pm.arima.ndiffs(y_train) 
D = pm.arima.nsdiffs(y_train, m = 48)

model = auto_arima(y_train,d = d, start_p=1, start_q=1, m= 48 ,D = 1, stepwise=True, trace=True, seasonal = True)

model.fit(y_train)
print(model.summary())
y_predict = model.predict(start=len(y_train), n_periods=len(x_test))
print(y_predict)
plt.plot(x_train,y_train, label = "training values")
plt.plot(x_test, y_predict, label = "prediction")
plt.plot(x_test,y_test, label = "test values")
plt.legend()
plt.show()




