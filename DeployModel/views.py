import joblib
import numpy as np
import keras
import pandas_datareader as web
from django.http import HttpResponse
from django.shortcuts import render
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def home(request):
    return render(request, "home.html")

def result(request):
    # return render(request, 'result.html')
    model = keras.models.load_model("saved_model.h5")
    scaler = MinMaxScaler(feature_range=(0, 1))

    startDate = request.GET['start_date']
    endDate = request.GET['end_date']
    data = web.DataReader('MSFT', data_source='yahoo', start= startDate, end= endDate)
    # new

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
    # get prediction
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    # print("Predicted price for next day: ", pred_price)


    # print(new)
    # pred = np.array(new.set_index("Close"))
    # p = pred[0][4]
    # p = np.array(p).reshape(1,1,1)
    # p


    
    # ans = scaler.predict(p)
    ans = pred_price.reshape(1)
    # print(ans)
    
    return render(request, "result.html", {'ans': ans})