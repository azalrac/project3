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
tickers = ['CMCSA', 'V', 'LMT', 'NVDA', 'FB', 'NFLX', 'FOX', 'ORCL', 'TSLA', 'BA']
names_dict = {'CMCSA':'Comcast', 'V':'Visa', 'LMT':'Lockheed Martin', 'NVDA':'Nivdia', 'FB':'Facebook', 'NFLX':'Netflix', 'FOX':'21st Century Fox', 'ORCL':'Oracle', 'TSLA':'Tesla', 'BA':'Boeing'}
sent_dict = {'CMCSA':"Comcast is the world's largest broadcasting and cable corporation. Comcast is the largest cable television and internet provider services. The company began in 1963 in Tupelo, Mississippi and went public in 1972 with an initial price of $7 per share. Comcast's stock price has risen steadily since it was initially offered and peaked for $42 a share in February 2018.", 
'V':'Visa yet another wonderful company', 
'LMT':"Lockheed Martin is a global security and aerospace company that employs approximately 100,000 people worldwide and is principally engaged in the research, design, development, manufacture, integration and sustainment of advanced technology systems, products and services. It was formed by the merger of Lockheed Corporation with Martin Marietta in March 1995. Lockheed Martin had $51 Billion in sales for 2017.", 
'NVDA':'Nvidia is an American technology company based in Santa Clara, California.  It designs graphics processing units (GPUs) for the gaming and professional markets, as well as system on a chip units (SoCs) for the mobile computing and automotive market. Its primary GPU product line, labeled GeForce, is in direct competition with Advanced Micro Devices (AMD) Radeon products. Nvidia expanded its presence in the gaming industry with its handheld Shield Portable, Shield Tablet and Shield Android TV.', 
'FB':'Facebook is a social media and networking website based in Menlo Park, California. Mark Zuckerberg and a group of Harvard students launched the platform in 2004, where it was limited to Harvard students. It eventually expanded its user base to other higher education schools around the Boston area and from there it was on its way to becoming the giant that it is today. Facebook applied for a $5 billion IPO in 2012 and opened with a sharing price of $38 per share. Today they are at the forefront of social media influence, acquiring Instagram in 2012 in addition to WhatsApp and Oculus VR.', 
'NFLX':"Netflix is an American media services provider, headquartered in Los Gatos, California. Founded in 1997 by Reed Hastings and Marc Randolph in Scotts Valley, California, the company's primary business is its subscription-based streaming media service, which offers online streaming of a library of films and television programs including those produced in-house.", 
'FOX':'Twenty-First Century Fox, Inc. is an American multinational mass media corporation that is based in Midtown Manhattan, New York City. It is one of the two companies formed from the 2013 spin-off of the publishing assets of News Corporation, as founded by Rupert Murdoch in 1979.', 
'ORCL':'Oracle Corporation is an American multinational computer technology corporation headquartered in Redwood Shores, California.', 
'TSLA': 'Tesla, Inc. is an American automotive and energy company based in Palo Alto, California. The company specializes in electric car manufacturing and, through its SolarCity subsidiary, in solar panel manufacturing. Tesla’s mission is to accelerate the world’s transition to sustainable energy',
'BA':'The Boeing Company is an American multinational corporation that designs, manufactures, and sells airplanes, rotorcraft, rockets, satellites, and missiles worldwide. The company also provides leasing and product support services.'
}
# identify the date interval
date1 = '2016-01-01'
date2 = str(date.today()) 

# adjclose is the same as close
# initialize empty list to append
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
    # print(f"Model accuracy for {ticker}: {accuracy}")
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
    return render_template("index.html", data = prediction_table, acc = acc_dic, act = actual_table, names = names_dict, sent = sent_dict)

# 2. Run App
if __name__ == '__main__':
    app.run(port = 5000, debug=True)