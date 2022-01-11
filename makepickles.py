import pickle
import datetime as dt
from datetime import date
import pandas as pd
import garbagefix
import predtopickle
import newclustavg
from garbagefix import FixGarbage
from newclustavg import NewClustAvg
from predtopickle import ArimaPred
from pullindexhist import *
from yfinpull import *
import numpy as np

def MakePickles():
    try:
        data = read_csv('csv/SnP500Close.csv')
    except:
        PullIndexHist
        data = read_csv('csv/SnP500Close.csv')
#retrieve today's date
    today = date.today()
    today = pd.to_datetime(today)


#convert date column to pandas datetime
    data['Date'] = pd.to_datetime(data['Date'])

#retreive latest date shown in historical data
    latest = data['Date'].iat[data['Date'].shape[0]-1]
#print(today,latest)

#compare latest date to current date, if older than 10 days, pull fresh historical data.
    timediff = today - latest
    timecap = dt.timedelta(days=10)
    print(timediff,timecap)

    if timediff > timecap:
        print('pulling new history for index fund')
        PullIndexHist
        NewClustAvg
        FixGarbage
    else:
        if timediff < timecap:
            print('historical data is still in date, continuing to clusters')
            NewClustAvg
            FixGarbage  
            for x in range(0,4):
                print(x)
                filename = '{}clustmodel'.format(x)
                print('pickling ',filename)
                data = read_csv('csv/{}_average.csv'.format(x))
                outfile = open(filename,'wb')
               
                model = ArimaPred(data)

                pickle.dump(model,outfile)

                outfile.close()


   

        



