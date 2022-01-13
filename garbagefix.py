import yfinance as yf
import pandas as pd
import numpy as np
from pandas import read_csv

def FixGarbage():
    dates = read_csv('csv/SnP500Close.csv')
    print(dates)
    dates.dropna(axis=1,inplace=True)
    date = dates['Date']
    for x in range(0,4):
        data = read_csv('csv/{}_average.csv'.format(x))
        data = data.iloc[:, 1:-2]
        data = data.T
        print(data.head(3),data.tail(3))
        print(data.shape,date.shape)
        data['Date'] = date.values
        data['Close'] = data[0]
        data = data.iloc[:,1:]
        data.to_csv('csv/{}_average.csv'.format(x))        



