import numpy as np
import pandas as pd
import joblib
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
plt.style.use('fivethirtyeight')


# Linear Regression


# df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')

# # show data
# df.head()

# # visualize the close price history
# plt.figure(figsize=(16, 8))
# plt.title('Close Price History')
# plt.plot(df['Close'])
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price ($)', fontsize=18)
# plt.show()


# X = df[['High', 'Low', 'Open', 'Volume', 'Adj Close']].to_numpy()
# y = df[['Close']].to_numpy()

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# reg = LinearRegression().fit(X_train, y_train)
# print("Score:")
# print(reg.score(X_train, y_train))
# print("Coefficients:")
# print(reg.coef_)
# print("Intercept:")
# print(reg.intercept_)


# df2 = web.DataReader('AAPL', data_source='yahoo', start='2021-03-30', end='2021-03-30')
# df2

# X_pred = df2[['High', 'Low', 'Open', 'Volume', 'Adj Close']].to_numpy()
# print(X_pred)
# y_pred = reg.predict(X_pred)
# y_pred



# LSTM

df = web.DataReader('MSFT', data_source='yahoo', start='2014-04-01', end='2021-04-01')
# df

plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price History USD ($)')
plt.show()

data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil( len(dataset) * 0.8 )

# print(training_data_len)


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# print(scaled_data)

train_data = scaled_data[0:training_data_len, :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])
  # if i<=61:
  #   print(x_train)
  #   print(y_train)
  #   print()
    
    
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# print(x_train.shape)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)


test_data = scaled_data[training_data_len-60: , :]

x_test = []
y_test = dataset[training_data_len: , :]

for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])
  

x_test = np.array(x_test)


x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
# print(predictions)


model.summary()

rmse = np.sqrt(np.mean(predictions-y_test)**2)
print("RMSE: ", rmse)



train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()



# print(valid)

apple_quote = web.DataReader('MSFT', data_source='yahoo', start='2014-01-01', end='2021-04-20')
new_df = apple_quote.filter(['Close'])
last_60_days = new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)

X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print("Predicted Price: ", pred_price)



new = web.DataReader('MSFT', data_source='yahoo', start='2021-04-21', end='2021-04-21')
print("------------------------------------------")
print("Printing prediction: ")
print(new)
print("------------------------------------------")


# Save model to disk
# filename = 'finalized_model.sav'
# joblib.dump(model, filename)

model.save("saved_model.h5")



















# def predict_price(to_predict_list):
#       to_predict = np.array(to_predict_list).reshape(1, 5)
#       price = reg.predict(to_predict)
#       return price


# result = predict_price([1.20403099e+02, 1.18860001e+02, 1.20110001e+02, 8.56719190e+07, 1.19900002e+02])
# print(result)

