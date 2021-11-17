import yfinance as yf
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import csv
import datetime
from sklearn import linear_model
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pandas import read_csv
from matplotlib import pyplot as plt


tickers=[
        'aapl',
        'msft'
    ]

#pulls historical close prices for tickers listed
#for ticker in tickers:
#    ticker_a = yf.Ticker(ticker)
#    data = ticker_a.history(period="max")
#    data['Close'].to_csv("close{}.csv".format(ticker))

#read the csv, from file
#obviously temp code, needs to be automated from the ground up to pull and train on ticker user selects

#reading in csv, skipping header row, adding own header to target for model
#data = read_csv("closeaapl.csv",skiprows=[0])
#data_head = ['date','close']

#im just gonna fucking keep the original head what the fuck even was that error
data = read_csv("closeaapl.csv")
rowcount = 185
data = data.tail(rowcount)



print(data['Date'].head(5))

for date in data['Date']:
    
    #convert string to epoch timestamp
    date = datetime.datetime.strptime(date, "%Y-%m-%d").timestamp()
    #date = date.apply(np(datetime.datetime.strptime(date, "%Y-%m-%d").timestamp()))    
    
print("data \n",data.tail(100))



#printing relevant dimension info that's here bc apparently it's important
#print(data.shape)
#print(data.head(5))
#print(data.tail(5))



target_head = ['Close']

X = data
y = data[target_head]
#print(data[target_head])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

y_pred_Train = model.predict(X_train)

y_pred = model.predict(X_test)

model_acc_train = accuracy_score(y_train, y_pred_train)

print("model accuracy on train set: {:.2f}\n".format(model_acc_train))
model_acc_test = accuracy_score(y_test, y_pred)
print("\n model accuracy on test set: {:.2f}\n".format(model_acc_test))




#plot the graph, DON'T PLOT BEFORE CUTTING OUT TRAIN/TEST, IT'S LITERALLY 40 YEARS OF DAILY DATA
#x = data['Date'].iloc[::15]
#y= data['Close'].iloc[::15]
#plt.plot(x,y)
#plt.title('test')
#plt.figure(1)
#plt.savefig('test.svg')



