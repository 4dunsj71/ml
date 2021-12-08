import os
import tkinter as tk
import yfinance as yf

pwd = '/home/james/documents/coursework/year3/machinelearning/ml'

def findFile(fileName):
    for root, dirs, files in os.walk(pwd):
        if fileName in files:
            print('found')
            return True
        else:
            print('not found')
            return False
            

def searchTicker():
    tickerName = tickerIn.get()
    ticker = yf.Ticker(tickerName)
    print(ticker.isin)
    fileStr = "close{}.csv".format(tickerName)
    print(fileStr)
    if  len(ticker.isin) < 4:
        tickerConf['text'] = "ticker not found, try again"
    else:
        if findFile(fileStr):
            print("Data already retrieved")
        else:
            data = ticker.history(period='300d', interval='1d')
            tickerConf['text'] = "ticker found and data retreived!"
            data['Close'].to_csv("close{}.csv".format(tickerName))
            
            

#def sub(var,entry):
#    print("button pressed")
#    var = entry.get()
#    print(var)

window = tk.Tk()

frameA = tk.Frame()
frameB = tk.Frame()
frameC = tk.Frame()
label = tk.Label(
        text="stock bot 3000",
        master = frameA)

tickerInLb = tk.Label(
        text="enter a valid ticker",
        master = frameB)

tickerIn = tk.Entry(
        text = "enter a valid ticker",
        width = 50,
        master = frameB)

tickerConf = tk.Label(text = "", master = frameC)

btn = tk.Button(
        text="search for stock",
        command=searchTicker,
        master = frameC
        )




frameA.pack()
frameB.pack()
frameC.pack()
label.pack()
tickerInLb.pack(side=tk.LEFT)
tickerIn.pack(side=tk.RIGHT)
tickerConf.pack(side=tk.LEFT)
btn.pack(side=tk.RIGHT)



window.mainloop()
