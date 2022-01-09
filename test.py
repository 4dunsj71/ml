import yfinance as yf
import pandas as pd
import numpy as np
from pandas import read_csv

def fixgarbage():
    dates = read_csv('csv/SnP500Close.csv')
    date = dates['Date']
    date = date.drop([0])

    for x in range(0,4):
        
        data = read_csv('csv/{}_average.csv'.format(x))
        data = data.iloc[:, 1:-2]
        data = data.T
        data['Date'] = date.values
        data['Close'] = data[0]
        data = data.iloc[:,1:]
        data.to_csv('csv/{}_average.csv'.format(x))        

    

