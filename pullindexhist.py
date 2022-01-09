import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
import csv
import datetime as dt
from pandas import read_csv
from numpy import log
from time import sleep

def PullIndexHist():

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

    for symbol in tickers:
        try:
            ticker = yf.Ticker(str(symbol))
            data = ticker.history(period='300d', interval='1d')
            sp500data[symbol] = data['Close']
            print('found:', symbol)    
        except:
            print("too many requests to yahoo finance, try again later")
        finally:
            sp500data.to_csv("csv/SnP500Close.csv")



