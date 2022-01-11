import pullindexhist
import newclustavg
import garbagefix
import yfinpull
import pandas as pd
import datetime as dt
from pullindexhist import PullIndexHist
from newclustavg import NewClustAvg
from garbagefix import FixGarbage
from yfinpull import YfinPull
from pandas import read_csv
from datetime import date, timedelta
from predfromclust import PredFromClust
#PullIndexHist)_
#NewClustAvg()
#FixGarbage()
data = read_csv('csv/closeaapl.csv')
data['Date'] = pd.to_datetime(data['Date']).dt.date
dates = data['Date']
prices =data['Close']
print(dates)
lastdate = data['Date'].values[-1]
lastprice = data['Close'].values[-1]

periods = 10

#lastdate = lastdate.astype('M8[D]')
pred = PredFromClust(data,periods)

lastpredplace = pred.size

td = timedelta(periods)

preddate = lastdate + td

print(lastdate,'\n',td,'\n',preddate)


#print(data.shape)
#data.dropna(axis =1,inplace=True)
#print(data.shape)
#data.drop(data.columns[0], axis=1, inplace=True)
#print(data.head(3))


