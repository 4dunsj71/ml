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
from time import sleep
from progress.bar import Bar
head = ['symbol',
        'security',
        'filings',
        'gics',
        'head',
        'location',
        'firstadd',
        'cik',
        'founded']


sp500 = read_csv("csv/s&p500.csv",skiprows=[0], names = head)
sp500['symbol'] = sp500['symbol'].str.replace(' ','')

tickers = sp500['symbol'].tolist()
print(tickers)

sp500data = pd.DataFrame()
with Bar('Retrieving data', fill='|') as bar:
    for symbol in tickers:
        try:
            ticker = yf.Ticker(str(symbol))
            data = ticker.history(period='300d', interval='1d')
            sp500data[symbol] = data['Close']
            print('found:', symbol)
            bar.next()    
        except:
            print("yahoo finance capped us laddy")
        finally:
            sp500data.to_csv("csv/SnP500Close.csv")


#pulls historical close prices for tickers listed
#for ticker in tickers:
#    ticker_a = yf.Ticker(ticker)
#    data = ticker_a.history(period='300d', interval='1d')
#    data['Close'].to_csv("close{}.csv".format(ticker))
#    print("saving close{}.csv".format(ticker))


