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


data = read_csv("SnP500OpenClose.csv")
data = data()
Open = np.array

Close = np.array

for symbol in data:
    openprice = data["{}open".format(symbol)]
    closeprice = data["{}close".format(symbol)]
    
    data["{}movement".format(symbol)] = openprice - closeprice

