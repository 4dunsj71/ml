import os
import tkinter as tk
import yfinance as yf
from yfinpull import *
from pred import *
from newclustavg import *
from pullindexhist import *

def searchTicker():
    tickerName = tickerIn.get()
    try:
        data = read_csv('csv/close{}.csv'.format(tickerName))
        if len(data) == 300:
            tickerConf['text'] = 'ticker found' 
    except:
        try:
            data = yfinpull(tickerName)
            tickerConf['text'] = 'ticker found on yfinance, retrieved.\n{}'
        except:
            tickerConf['text'] = 'Ticker not found, check spelling'
 
def makepred():
    tickerName = tickerIn.get()
    try:
         print(tickerName)
         prediction = ArimaPred(tickerName)
         predout['text'] = prediction
         tickerConf['text'] = 'please wait for prediction:'
    except:
        tickerConf['text'] = 'an error has occured, please search for the ticker again'

window = tk.Tk()

frameA = tk.Frame()
frameB = tk.Frame()
frameC = tk.Frame()
frameD = tk.Frame()
frameE = tk.Frame()
label = tk.Label(
        text="stock bot 3000",
        master = frameA)

tickerInLb = tk.Label(
        text="enter a valid ticker",
        master = frameB)

tickerIn = tk.Entry(
        text = "enter a valid ticker",
        width = 20,
        master = frameB)

tickerConf = tk.Label(text = "", master = frameC)

btn = tk.Button(
        text="search for stock",
        command=searchTicker,
        master = frameD
        )

submitpred = tk.Button(
        text = 'make prediction',
        command = makepred,
        master = frameD
        )

predout = tk.Label(
        text = '',
        master = frameE
        )

frameA.pack()
frameB.pack()
frameC.pack()
frameD.pack()
frameE.pack()
label.pack()
tickerInLb.pack(side=tk.LEFT)
tickerIn.pack(side=tk.RIGHT)
tickerConf.pack(side=tk.LEFT)
btn.pack(side=tk.LEFT)
submitpred.pack(side=tk.RIGHT)
predout.pack()


window.mainloop()
