from django.http import HttpResponse
from django.shortcuts import render
import joblib

def home(request):
    return render(request, "home.html")

def result(request):
    # return render(request, 'result.html')
    scaler = joblib.load('finalized_model.sav')

    lis = []
    
    lis.append('MSFT')
    lis.append('yahoo')
    lis.append(request.GET['start_date'])
    lis.append(request.GET['end_date'])  
        
    # lis.append(request.GET['high'])
    # lis.append(request.GET['low'])
    # lis.append(request.GET['open']) 
    # lis.append(request.GET['volume'])
    # lis.append(request.GET['adj'])
    
    print("Printing list ---->   ")
    print(lis)
    
    
    ans = scaler.predict([lis])
    
    return render(request, "result.html", {'ans': ans})