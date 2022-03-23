import numpy as np
import pandas as pd
import pickle
import keras
import re
import pandas_datareader as web
from django.http import HttpResponse
from django.shortcuts import render
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from newsapi import NewsApiClient
from datetime import timedelta

def index(request):
    return render(request, "index.html")

def invTransform(scaler, data, colName, colNames):
    dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
    dummy[colName] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
    return dummy[colName].values

def predict_price(company_tag, start_date, end_date):
    #load models
    infy_price_model = keras.models.load_model('.\model\infy.h5')
    scaler = pickle.load(open('.\model\scaler.pkl', 'rb'))
    
    #fetch data of specified period
    try:
        df = web.DataReader(company_tag, data_source='yahoo', start=start_date, end=end_date)
    except:
        return -1
    
    new_df = df.filter(items = ['High', 'Low', 'Open', 'Close', 'Volume'])
    previous_value = new_df.iloc[-1, 3]
    
    #predict price using last 60-day bunch
    last_60_days = new_df[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = []
    X_test.append(last_60_days_scaled[:, np.array([True, True, True, False, True])])
    X_test = np.array(X_test)
    global pred_price
    pred_price = infy_price_model.predict(X_test)
    pred_price = invTransform(scaler, pred_price, 'Close', ['High', 'Low', 'Open', 'Close', 'Volume'])
    pred_price = pred_price[0]

    #compare pred_price with last available stock price
    #if price increases -> 1, decreasing -> -1
    global price_change
    price_change = 0
    if(pred_price > previous_value):
        price_change = 1
    elif(pred_price < previous_value):
        price_change = -1

def predict_sentiment(company_name, start_date, end_date, rg):
    #load models
    infy_sentiment_model = pickle.load(open('.\model\sentiment.sav', 'rb'))
    countvector = pickle.load(open('.\model\count_vectorizer.pkl', 'rb'))
  
    #get all articles related to the company
    newsapi = NewsApiClient(api_key='34b0879255f547818b431e10afab8641')
    try:
        all_articles = newsapi.get_everything(q=company_name, from_param=start_date, to=end_date,
        language='en')
    except:
        return -1

    n = len(all_articles.get('articles'))
    news = []
    
    #add each news to array which is relevant with regex
    for i in range(n):
        x = re.search(rg, all_articles.get('articles')[i].get('title'), re.IGNORECASE)
        if(x):
            news.append(all_articles.get('articles')[i].get('title'))
    
    #convert news array into dataframe
    df_news = pd.DataFrame(news, columns=['News'])
    
    #get sentiments on news statements
    if(len(news) != 0):
        news_dataset = countvector.transform(news)
        predictions = infy_sentiment_model.predict(news_dataset)
        df_news['Sentiment'] = predictions
    
    #get the average sentiment from news headlines
    positive_sentiment = 0
    for i in df_news.index:
        if(df_news['Sentiment'][i] == 1):
            positive_sentiment += 1
    
    #return the average sentiment
    global sentiment_change
    if(positive_sentiment > df_news.shape[0]/2):
        sentiment_change = 1
    else:
        sentiment_change = 0

def result(request):  
    # fetch required data
    company_name = request.GET['company-name']
    date = request.GET['date'] #current date

    company_tag = ''
    if(company_name == 'Infosys'):
        company_tag = 'INFY'
    elif(company_name == 'Tata Motors'):
        company_tag = 'TTM'
    else:
        company_tag = 'ACC'

    #get date of 3 months before
    subtracted_date = pd.to_datetime(date) - timedelta(weeks=15)
    subtracted_date = subtracted_date.strftime("%Y-%m-%d")
    #call function to predict price using last 60 days
    if(predict_price(company_tag, subtracted_date, date) == -1):
        return render(request, "result.html", {'ans': 'Unable to fetch data'})    

    #get date of 3 days before
    subtracted_date = pd.to_datetime(date) - timedelta(days=3)
    subtracted_date = subtracted_date.strftime("%Y-%m-%d")
    #call function to predict sentiment of news with last 3 days
    if(predict_sentiment(company_name, subtracted_date, date, company_name) == -1):
        return render(request, "result.html", {'ans': 'Unable to fetch data'})
    
    #get required result according to both predictions
    if(sentiment_change == 1):
        ans = pred_price
    else:
        if(price_change == 1):
            ans = pred_price
        else:
            ans = 'Price will decrease'
    
    return render(request, "result.html", {'ans': ans})