import tkinter as tk
import pandas as pd
import numpy as np
import yfinpull
import pred
import newclustavg
import pullindexhist
import compare
import predfromclust
import datetime as dt
from pandas import read_csv
from yfinpull import YfinPull
from pred import ArimaPred
from newclustavg import NewClustAvg
from pullindexhist import PullIndexHist
from compare import Compare
from predfromclust import PredFromClust
from datetime import date, timedelta
def searchforticker():
    tickername = tickerIn.get()
    if tickername != "":
        try:
            data = read_csv('csv/close{}.csv'.format(tickername))
            if len(data) == 300:
                tickerConf['text'] = 'ticker found in storage'
                #print(data['Close'])
        except:
            YfinPull(tickername)
            data = read_csv('csv/close{}.csv'.format(tickername))
            #print(data['Close'])
            if len(data) == 300:
                data.to_csv('csv/close{}.csv'.format(tickername))
                tickerConf['text'] = 'ticker history retrieved from yfinance'
    else:
        tickerConf['text'] = 'an error has occured, please check the ticker spelling'

def makeprediction():
    tickername = tickerIn.get()
    periods = int(periodsIn.get()) + 60
    if tickername !="":
        try:
            data = read_csv('csv/close{}.csv'.format(tickername))
            if data.shape[1] == 3:
                data.drop(data.columns[0],axis = 1, inplace=True)
            
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
            if diff[1]<20:
                print(diff)
                prediction = PredFromClust(data,periods)
                lastpred = prediction[-1]
                print(lastpred)
                predConf['text'] = 'prediction date:{}\nprediction = £:{}\nlast date in system:{}\nlast price in system:{}'.format(preddate,round(lastpred,2),lastdate,round(lastprice,2))
            else:
                prediction = ArimaPred(data,periods)
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
tickerIn.pack(side=tk.LEFT)
periodsIn.pack(side=tk.RIGHT)
tickerSearch.pack(side=tk.LEFT)
makePred.pack(side=tk.RIGHT)
tickerConf.pack(side=tk.LEFT)
predConf.pack(side=tk.LEFT)

window.mainloop()
