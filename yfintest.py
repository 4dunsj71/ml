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
        'aapl',
        'msft'
    ]

#pulls historical close prices for tickers listed
#for ticker in tickers:
#    ticker_a = yf.Ticker(ticker)
#    data = ticker_a.history(period='300d', interval='1d')
#    data['Close'].to_csv("close{}.csv".format(ticker))

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

x = data['Date'].values
y = data['Close'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

print(x_train.shape, "\n", x_test.shape)

lr_regress= LinearRegression()
lr_regress.fit(x_train, y_train)
y_linear_pred = lr_regress.predict(x_test)

#SEED = 1

#dt_model = DecisionTreeRegressor(max_depth = 8, min_samples_leaf= 0.13, random_state=1)

#dt_model.fit(x_train, y_train)
#y_pred = dt_model.predict(x_test)
#y_pred_train = dt_model.predict(x_train)


poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(x_train)
x_poly_test = poly_reg.fit_trainsform(x_test)
poly_reg.fit(X_poly,y_train)
y_pred = poly_reg.predict(X_poly_test)

lr_regress= LinearRegression()
lr_regress.fit(x_train, y_train)
y_linear_pred = lr_regress.predict(x_test)


mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

#print('\n slope:', lr_regress.coef_, '\nIntercept: ', lr_regress.intercept_, '\nMean absolute error: {:.2f}'.format(mae), '\nmean_squared_error: {:.2f}'.format(mse), '\nroot mean squared error: {:.2f}'.format(rmse), '\nR2 score: ', r2)

#print('\n slope:', dt_model.coef_, '\nIntercept: ', dt_model.intercept_, '\nMean absolute error: {:.2f}'.format(mae), '\nmean_squared_error: {:.2f}'.format(mse), '\nroot mean squared error: {:.2f}'.format(rmse), '\nR2 score: ', r2)


plt.scatter(x_test, y_test, s=10)
plt.xlabel('x - time')
plt.ylabel('y - price')
plt.plot(x_test, y_pred, color='r')
plt.figure(1)
plt.savefig('polynomtest.svg')




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



