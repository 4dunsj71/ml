import yfinance as yf
import pandas as pd
import numpy as np
import csv
from pandas import read_csv


def yfinpull(ticker): 
    tick = yf.Ticker(ticker)
    data = tick.history(period='300d', interval='1d')
    data['Close'].to_csv("close{}.csv".format(ticker))
    

