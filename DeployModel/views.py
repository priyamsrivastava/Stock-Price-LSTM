import joblib
import numpy as np
import keras
import pandas_datareader as web
from django.http import HttpResponse
from django.shortcuts import render
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def index(request):
    return render(request, "index.html")

def result(request):
    # load all the models
    apple_model = keras.models.load_model(".\pple.h5")
    google_model = keras.models.load_model(".\goog.h5")
    microsoft_model = keras.models.load_model(".\msft.h5")
    
    # for scaling the testing dataset
    scaler = MinMaxScaler(feature_range=(0, 1))

    # get data for testing
    company_name = request.GET['company-name']
    date = request.GET['date']

    # get respective data of company
    if(company_name == 'apple'):
        data = web.DataReader('AAPL', data_source='yahoo', start= '26-04-2020', end= date)
    elif(company_name == 'microsoft'):
        data = web.DataReader('MSFT', data_source='yahoo', start= '26-04-2020', end= date)
    else:
        data = web.DataReader('GOOG', data_source='yahoo', start= '26-04-2020', end= date)
    # filter the data to get closing price ot of whole dataset
    data_filtered = data.filter(['Close'])
    
    # form data for scaling
    dataset = data_filtered.values
    scaler.fit_transform(dataset)
    
    # take out last 60 days dataset for prediction
    last_60_days = data_filtered[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)
    
    # form a test dataset
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    
    # reshape dataset for deep learning LSTM model
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # get prediction with respective model
    if(company_name == 'apple'):
        pred_price = apple_model.predict(X_test)
    elif(company_name == 'microsoft'):
        pred_price = microsoft_model.predict(X_test)
    else:
        pred_price = google_model.predict(X_test)
    # inverse the scaling
    pred_price = scaler.inverse_transform(pred_price)
    
    # return predicted price to frontend
    ans = pred_price.reshape(1)[0]
    return render(request, "result.html", {'ans': ans})