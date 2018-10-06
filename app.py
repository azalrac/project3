# -*- coding: utf-8 -*-
import pandas as pd
from flask import Flask, jsonify, render_template 
from yahoofinancials import YahooFinancials
import numpy as np
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# identify how many days into the future do we want to predict
future = int(30)
# identify the stocks
#tickers = ['AAPL', 'MSFT', 'AMZN', 'CAT', 'NVDA', 'V', 'PYPL', 'LMT', 'GOOG', 'TXN']
#tickers = ['AAPL', 'DIS', 'FOX', 'CAT', 'NVDA', 'V', 'CMCSA', 'LMT', 'GOOG', 'TSLA']
#tickers = ['HD', 'FB', 'JNJ', 'KO', 'MDT', 'NFLX', 'ORCL', 'SBUX', 'WMT', 'XOM']
tickers = ['ABT', 'AGN', 'BA', 'COF', 'GOOG', 'GOOGL', 'CSCO', 'F', 'GE', 'LOW']
# identify the date interval
date1 = '2016-01-01'
date2 = str(date.today()) 

# adjclose is the same as close
# Initialize empty list to append
ti = []
acc = []
pred = []
act = []
for ticker in tickers:
    dat = pd.DataFrame()
    yahoo_financials = YahooFinancials(ticker)
    result = yahoo_financials.get_historical_price_data(date1, date2, 'daily')
    df = pd.DataFrame(data=result[ticker]['prices'])
    df['ticker'] = ticker
    df.drop(columns=['close', 'date'], inplace=True)
    df.rename(columns={'formatted_date':'date'}, inplace=True)
    dat = pd.concat([dat, df], ignore_index=True)
    dat = dat[['ticker', 'date', 'open', 'adjclose', 'low', 'high', 'volume']]
    dat = dat[['adjclose']]
    # predicting the last n days
    act.append(dat['adjclose'].values[-future:])
    dat['prediction'] = dat[['adjclose']].shift(-future)
    
    # prepare X and y
    X = np.array(dat.drop(['prediction'], axis=1))
    X = preprocessing.scale(X)
    # set forecast to the last n rows
    X_forecast = X[-future:] 
    # get rid of the last n nan rows
    X = X[:-future]
    # get y
    y = np.array(dat['prediction'])
    y = y[:-future]
    
    # train test split 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    
    # linear Regression
    clf = LinearRegression()
    clf.fit(X_train,y_train)
    # Testing
    accuracy = clf.score(X_test, y_test)
    #print(f"Model accuracy for {ticker}: {accuracy}")
    forecast = clf.predict(X_forecast)
    
    ti.append(ticker)
    acc.append(accuracy)
    pred.append(forecast)
    
df = pd.DataFrame(pred).T
df.columns = ti
dates = pd.date_range(end=pd.datetime.today(), periods=future).date.astype('str').tolist()
df['Date'] = dates

actual = pd.DataFrame(act).T
actual.columns = ti
dates = pd.date_range(end=pd.datetime.today(), periods=future).date.astype('str').tolist()
actual['Date'] = dates

acc_dic = {}
for i in range(10):
    acc_dic[tickers[i]] = acc[i]

print('data ready!')
    
# 0. Create app
app = Flask(__name__)

# 0. Initialize content page
# @app.route('/')
# def Welcome():
#     return (
#         f"Welcome to this API!<br/>"
#         f"<br/>"
#         f"Available Routes:<br/>"
#         f"<br/>"
#         f"/api/v1.0/prediction<br/>"
#         f"/api/v1.0/accuracy<br/>"
#         f"/api/v1.0/actuals<br/>"
# #        f"/api/v1.0/start<br/>"
# #        f"/api/v1.0/start/end"
#         )

# 1. prediction
@app.route('/')
def Prediction():
    prediction_table = []
    for i in range(df.shape[0]):
        dic = {}
        for j in df.columns:
            dic[j] = df[j][i]
        prediction_table.append(dic)

    actual_table = []
    for i in range(actual.shape[0]):
        dic = {}
        for j in actual.columns:
            dic[j] = actual[j][i]
        actual_table.append(dic)
    return render_template("index.html", data = prediction_table, acc = acc_dic, act = actual_table)

# 5. Run App
if __name__ == '__main__':
    #app.run(port = 5000, debug=True)
    app.run()