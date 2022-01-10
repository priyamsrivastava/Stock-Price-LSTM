# import necessary libraries
import numpy as np
import pandas as pd
import math
import pandas_datareader as web
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

# get the dataset
df = web.DataReader('INFY', data_source='yahoo', start='2015-01-01', end='2021-12-31')

# filter relevant data for model
data = df.filter(items = ['High', 'Low', 'Open', 'Close', 'Volume'])
dataset = data.values
training_data_len = math.ceil( len(dataset) * 0.8 )

# scale the data
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(dataset)
train_data = scaled_data[0:training_data_len, :]

x_train = []
y_train = []
for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, :])
  y_train.append(train_data[i, 3])

# sequential model needs numpy array as input
x_train, y_train = np.array(x_train), np.array(y_train)

# build and train model
model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 5)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=5)

# test model
test_data = scaled_data[training_data_len-60: , :]
x_test = []
y_test = dataset[training_data_len: , 3]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, :])  

x_test, y_test = np.array(x_test), np.array(y_test)

def invTransform(scaler, data, colName, colNames):
  dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
  dummy[colName] = data
  dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
  return dummy[colName].values

predictions = model.predict(x_test)
predictions = invTransform(scaler, predictions, 'Close', ['High', 'Low', 'Open', 'Close', 'Volume'])