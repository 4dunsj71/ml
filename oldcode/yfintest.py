import yfinance as yf
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import csv
import datetime as dt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, confusion_matrix, classification_report

tickers=[
        
        'aapl'
    ]

#pulls historical close prices for tickers listed
for ticker in tickers:
    ticker_a = yf.Ticker(ticker)
    data = ticker_a.history(period='900d', interval='1d')
    data['Close'].to_csv("close{}.csv".format(ticker))

#read the csv, from file
#obviously temp code, needs to be automated from the ground up to pull and train on ticker user selects

#reading in csv, skipping header row, adding own header to target for model
#data = read_csv("closeaapl.csv",skiprows=[0])
#data_head = ['date','close']

#im just gonna fucking keep the original head what the fuck even was that error

data = read_csv("closeaapl.csv")
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].astype('int64').astype(float)
data['Close'] = data['Close'].astype(float)

#print("date:\n",data['Date'].tail(10),"\nclose:\n",data['Close'].tail(10))



train_head = ['Date']
target_head = ['Close']


X = data[train_head]
y = data[target_head]
#print(X,"\n", y)

x = data['Date'].values.reshape(-1, 1)
y = data['Close'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


poly_transform = PolynomialFeatures(degree=2)

x_poly_train = poly_transform.fit_transform(x_train)
x_poly_test = poly_transform.fit_transform(x_test)

linear_reg = LinearRegression()

linear_reg.fit(x_poly_train, y_train)


x_grid_train = np.arange(min(x_train),max(x_train),0.1)
x_grid_test = np.arange(min(x_test),max(x_test),0)

x_grid_train = x_grid_train.reshape(len(x_grid_train), 1)
x_grid_test = x_grid_test.reshape(len(x_grid_test), 1)

#x_grid_train = x_train.reshape(len(x_train), 1)
#x_grid_test = x_test.reshape(len(x_test), 1)




#mae = mean_absolute_error(y_test,y_pred)
#mse = mean_squared_error(y_test,y_pred)
#rmse = np.sqrt(mse)
#r2 = r2_score(y_test,y_pred)

plt.scatter(x, y, color='blue', label='Training Data')
#plt.scatter(x_test, y_test, label ='data' , color = 'blue')
plt.plot(x_grid_train, linear_reg.predict(poly_transform.fit_transform(x_grid_train)), color='red', label='model curve')
plt.xlabel('x - time')
plt.ylabel('y - price')
#plt.plot(x,pol_reg.predict(poly_reg.fit_transform(x)), color='red')
plt.legend()
plt.show()
#plt.figure(1)
#plt.savefig('polynomtest.svg')




#plt.scatter(x_test, y_test, s=10)
#plt.xlabel('x - time')
#plt.ylabel('y - price')
#plt.plot(x_test, y_pred, color='r')
#plt.figure(1)
#plt.savefig('decisiontreetest1.svg')

#-----------------------------------------------------------------------------------------------------------------------------


#model = linear_model.LinearRegression()
#model.fit(X_train, y_train)

#y_pred_train = model.predict(X_train)

#y_pred = model.predict(X_test)

#model_acc_train = accuracy_score(y_train, y_pred_train)

#print("model accuracy on train set: {:.2f}\n".format(model_acc_train))
#model_acc_test = accuracy_score(y_test, y_pred)
#print("\n model accuracy on test set: {:.2f}\n".format(model_acc_test))


#plot the graph, DON'T PLOT BEFORE CUTTING OUT TRAIN/TEST, IT'S LITERALLY 40 YEARS OF DAILY DATA
#x = data[train_head]
#y= data[target_head]
#plt.plot(x,y)
#plt.title('test')
#plt.figure(1)
#plt.savefig('test2.svg')



