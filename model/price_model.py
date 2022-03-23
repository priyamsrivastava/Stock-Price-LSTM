#make necessary imports
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pickle

# fetch dataset from yahoo finances
df = web.DataReader('ACC', data_source='yahoo', start='2016-01-01', end='2022-02-28')

#filter relevant columns from dataframe
data = df.filter(items = ['High', 'Low', 'Open', 'Close', 'Volume'])
#convert data to numpy array
dataset = data.values
#80% of dataset will be used for training
training_data_len = math.ceil( len(dataset) * 0.8 )

#scale the dataset for uniformity between 0 and 1
scaler = MinMaxScaler(feature_range = (0, 1)) #initialize an scaler instance
scaled_data = scaler.fit_transform(dataset)

#get the training dataset
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []
#columns except 'Close' are in x_train and result in y_train
for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, np.array([True, True, True, False, True])])
  y_train.append(train_data[i, 3])

#convert to numpy array
x_train, y_train = np.array(x_train), np.array(y_train)

#initialize a sequential model with 4 layers 
model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 4)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

#compile and train model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=5)

#get the testing dataset
test_data = scaled_data[training_data_len-60: , :]
x_test = []
y_test = dataset[training_data_len: , 3]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, np.array([True, True, True, False, True])])

#convert it to numpy array
x_test, y_test = np.array(x_test), np.array(y_test)

#function to reverse scale the data
def invTransform(scaler, data, colName, colNames):
    dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
    dummy[colName] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
    return dummy[colName].values

#predict on x_test and get the non-scaled prediction value
predictions = model.predict(x_test)
predictions = invTransform(scaler, predictions, 'Close', ['High', 'Low', 'Open', 'Close', 'Volume'])

#print the RMSE score of the model
rmse = np.sqrt(np.mean(predictions-y_test)**2)
print(rmse)