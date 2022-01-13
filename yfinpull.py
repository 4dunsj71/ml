import yfinance as yf
import pandas as pd
import numpy as np
import csv
from pandas import read_csv


def YfinPull(ticker):
    ticker = str(ticker)
    tick = yf.Ticker(ticker)
    data = tick.history(period='900d', interval='1d')
    data['Close'].to_csv("csv/close{}.csv".format(ticker))    






