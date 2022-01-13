import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib
from makepickles import MakePickles
import yfinpull
import pred
import newclustavg
import pullindexhist
import compare
import predfromclust
import garbagefix
import makepickles
import datetime as dt
from pandas import read_csv
from yfinpull import YfinPull
from pred import ArimaPred
from newclustavg import NewClustAvg
from pullindexhist import PullIndexHist
from compare import Compare
from predfromclust import PredFromClust
from garbagefix import FixGarbage
from makepickles import MakePickles
from datetime import date, timedelta
from matplotlib import pyplot as plt

#check for historical data and check it's in date
try:
    data = read_csv('csv/SnP500Close.csv')
except:
    PullIndexHist()
    data = read_csv('csv/SnP500Close.csv')
today = date.today()
today = pd.to_datetime(today)
snpdates = pd.to_datetime(data['Date'])
latestsnp = snpdates.iat[snpdates.shape[0]-1]
timediff = today - latestsnp
timecap = dt.timedelta(days = 30)

#check for cluster averages
try:
    data = read_csv('csv/0_average.csv')
except:
    NewClustAvg()
    FixGarbage()
#check for pretrained models
try:
    infile = open('0clustmodel','rb')
    newmod = pickle.load(infile)
    infile.close()
except:
    MakePickles()

if timediff > timecap:
    NewClustAvg()
    FixGarbage()
    MakePickles()

        

def searchforticker():
    
    #historylength = int(historyIn.get())
    tickername = tickerIn.get()
    if tickername != "":
        try:
            data = read_csv('csv/close{}.csv'.format(tickername))
            if len(data) != 0:
                tickerConf['text'] = 'ticker found in storage'
                #print(data['Close'])
        except:
            YfinPull(tickername)
            data = read_csv('csv/close{}.csv'.format(tickername))
            #print(data['Close'])
            if len(data) != 0:
                data.to_csv('csv/close{}.csv'.format(tickername))
                tickerConf['text'] = 'ticker history retrieved from yfinance'
    else:
        tickerConf['text'] = 'an error has occured, please check the ticker spelling'

def makeprediction():
    tickername = tickerIn.get()
    periods = int(periodsIn.get())
    if tickername !="":
        try:
            data = read_csv('csv/close{}.csv'.format(tickername))
            if data.shape[1] == 3:
                data.drop(data.columns[0],axis = 1, inplace=True)
                print(data)
            data['Date'] = pd.to_datetime(data['Date']).dt.date
            dates = data['Date']
            prices =data['Close']
            lastdate = data['Date'].values[-1]
            lastprice = data['Close'].values[-1]
            td = timedelta(periods)
            preddate = lastdate + td
            print(lastprice,lastdate)
            
            diff = Compare(data)
            print(diff)
            if diff[1]<50:
                print(diff)
                prediction = PredFromClust(data,periods)
                print(prediction)
                newdates = pd.Series([])
                for x in range(0,periods):
                    datediff = timedelta(x)
                    newdate = lastdate + datediff
                    newdates = np.append(newdates,newdate)
                xaxis = data['Date'].tail(10)
                xaxis2 = newdates
                yaxis = data['Close'].tail(10)
                yaxis2 = round(prediction,2)

                plt.plot(xaxis,yaxis,label = 'data from yfinance')
                plt.plot(xaxis2,yaxis2,label = 'predictions')
                plt.legend()
                plt.show()
                lastpred = prediction[-1]
                print(lastpred)
                predConf['text'] = 'prediction date:{}\nprediction = £:{}\nlast date in system:{}\nlast price in system:{}'.format(preddate,round(lastpred,2),lastdate,round(lastprice,2))
            else:
                prediction = ArimaPred(data,periods)
                newdates = pd.Series([])
                for x in range(0,periods):
                    datediff = timedelta(x)
                    newdate = lastdate + datediff
                    newdates = np.append(newdates,newdate)
                xaxis = data['Date']
                xaxis = xaxis.tail(10)
                xaxis2 = newdates
                yaxis = data['Close']
                yaxis = yaxis.tail(10)
                yaxis2 = round(prediction,2)

                plt.plot(xaxis,yaxis,label = 'data from yfinance')
                plt.plot(xaxis2,yaxis2,label = 'predictions')
                plt.legend()
                plt.show()

                lastpred = prediction[-1]
                predConf['text'] = 'prediction date:{}\nprediction = £:{}\nlast date in system:{}\nlast price in system:{}'.format(preddate,round(lastpred,2),lastdate,round(lastprice,2))
        except:
            tickerConf['text'] = 'an error has occured, please search for the ticker and try again'



window = tk.Tk()

frame1 = tk.Frame()
frame2 = tk.Frame()
frame3 = tk.Frame()
frame4 = tk.Frame()
frame5 = tk.Frame()
title = tk.Label(
        text='stock predictor',
        master=frame1
        )
historyIn = tk.Entry(
        text = 'input a length of history in days',
        width = 5,
        master = frame1
        )
tickerIn = tk.Entry(
        text = '',
        width = 10,
        master=frame2

        )

periodsIn = tk.Entry(
        text = '',
        width = 10,
        master=frame2

        )

tickerSearch = tk.Button(
        text = 'Search for Stock',
        command = searchforticker,
        master=frame3

        )

makePred = tk.Button(
        text='make a prediction',
        command = makeprediction,
        master = frame3

        )

tickerConf = tk.Label(
        text = '',
        master=frame4

        )
predConf = tk.Label(
        text = '',
        width=100,
        height=50,
        master=frame5

        )
frame1.pack()
frame2.pack()
frame3.pack()
frame4.pack()
frame5.pack()
title.pack()
#historyIn.pack(side=tk.RIGHT)
tickerIn.pack(side=tk.LEFT)
periodsIn.pack(side=tk.RIGHT)
tickerSearch.pack(side=tk.LEFT)
makePred.pack(side=tk.RIGHT)
tickerConf.pack(side=tk.LEFT)
predConf.pack(side=tk.LEFT)

window.mainloop()
