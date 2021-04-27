import joblib
import numpy as np
import keras
import pandas_datareader as web
from django.http import HttpResponse
from django.shortcuts import render
from keras.models import load_model

def home(request):
    return render(request, "home.html")

def result(request):
    # return render(request, 'result.html')
    scaler = keras.models.load_model("saved_model.h5")

    startDate = request.GET['start_date']
    endDate = request.GET['end_date']
    new = web.DataReader('MSFT', data_source='yahoo', start= startDate, end= endDate)
    new

    print(new)
    pred = np.array(new.set_index("Close"))
    p = pred[0][4]
    p = np.array(p).reshape(1,1,1)
    p
    
    ans = scaler.predict(p)
    ans = ans.reshape(1)
    print(ans)
    
    return render(request, "result.html", {'ans': ans})