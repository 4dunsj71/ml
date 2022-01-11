import pullindexhist
import newclustavg
import garbagefix
import yfinpull
import pred
import pandas as pd
import datetime as dt
from pullindexhist import PullIndexHist
from newclustavg import NewClustAvg
from garbagefix import FixGarbage
from yfinpull import YfinPull
from pandas import read_csv
from datetime import date, timedelta
from predfromclust import PredFromClust
from pred import ArimaPred
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

periods = 100

#lastdate = lastdate.astype('M8[D]')
pred = ArimaPred(data,periods)

#lastpredplace = pred.size

#td = timedelta(periods)

#preddate = lastdate + td

#print(lastdate,'\n',td,'\n',preddate)


#print(data.shape)
#data.dropna(axis =1,inplace=True)
#print(data.shape)
#data.drop(data.columns[0], axis=1, inplace=True)
#print(data.head(3))
newdates = pd.Series()
for x in range(0,periods):
    datediff = timedelta(x)
    newdate = lastdate + datediff
    newdates[x] = newdate
xaxis = data['Date']
xaxis2 = newdates
yaxis = data['Close']
yaxis2 = prediction

plt.plot(xaxis,yaxis,label = 'data from yfinance')
plt.plot(xaxis2,yaxis2,label = 'predictions')
plt.legend()
plt.show()

prices = data.tail(100)['Close']
pred = pred.slice(-10)
print(prices,'\n',pred)
