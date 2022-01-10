import numpy as np
import pandas as pd
import keras
import pandas_datareader as web
from django.http import HttpResponse
from django.shortcuts import render
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def index(request):
    return render(request, "index.html")

def invTransform(scaler, data, colName, colNames):
    dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
    dummy[colName] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
    return dummy[colName].values

def result(request):
    # load all the models
    infy_model = keras.models.load_model(".\infy.h5")
    ttm_model = keras.models.load_model(".\motors.h5")
    acc_model = keras.models.load_model(".\cem.h5")
    
    # for scaling the testing dataset
    scaler = MinMaxScaler(feature_range=(0, 1))

    # fetch required data
    company_name = request.GET['company-name']
    date = request.GET['date']

    # get respective data of company
    if(company_name == 'Infosys'):
        data = web.DataReader('INFY', data_source='yahoo', start= '01-10-2021', end= date)
    elif(company_name == 'Tata Motors'):
        data = web.DataReader('TTM', data_source='yahoo', start= '01-10-2021', end= date)
    else:
        data = web.DataReader('ACC', data_source='yahoo', start= '01-10-2021', end= date)
    
    # filter the data to get closing price ot of whole dataset
    data_filtered = data.filter(items = ['High', 'Low', 'Open', 'Close', 'Volume'])
    
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
    
    # get prediction with respective model
    if(company_name == 'Infosys'):
        pred_price = infy_model.predict(X_test)
    elif(company_name == 'Tata Motors'):
        pred_price = ttm_model.predict(X_test)
    else:
        pred_price = acc_model.predict(X_test)
    
    # inverse the scaling
    pred_price = invTransform(scaler, pred_price, 'Close', ['High', 'Low', 'Open', 'Close', 'Volume'])
    
    # return predicted price to frontend
    ans = pred_price.reshape(1)[0]
    return render(request, "result.html", {'ans': ans})