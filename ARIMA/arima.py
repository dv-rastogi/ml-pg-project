# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tqdm
import glob
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive', force_remount = True)

fileName = glob.glob("/content/drive/MyDrive/Stock/CNNpred/Processed_NYSE.csv")[0]
df = pd.read_csv(fileName, lineterminator='\n')
df

import datetime

def make_datetime(date_str):
  return datetime.datetime.strptime(date_str, "%Y-%m-%d")

df['Date'] = df['Date'].apply(make_datetime)

df = df.sort_values('Date')
plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Closing Price History')

from statsmodels.tsa.stattools import adfuller

from pandas.plotting import autocorrelation_plot

from pandas.plotting import autocorrelation_plot

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm

"""# Splitting Data"""

import math

df_log = df['Close']
#split data into train and training set
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(df_log, 'green', label='Train data')
plt.plot(test_data, 'blue', label='Test data')
plt.legend()

#!pip3 install pmdarima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima

from sklearn.metrics import mean_squared_error, mean_absolute_error

model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0, test='adf', max_p=100, max_q=100, m=1, d=None, seasonal=False, start_P=0, D=0)
print(model_autoARIMA.summary())
model_autoARIMA.plot_diagnostics(figsize=(15,8))
plt.show()

"""Best model is with parameters (0, 1, 0)"""

model = ARIMA(train_data, order=(0,1,0))  
fitted = model.fit(disp=-1)
print(fitted.summary())

fc, se, conf = fitted.forecast(199, alpha=0.05)
fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)

plt.figure(figsize=(10,5), dpi=100)
plt.plot(train_data, label='training data')
plt.plot(test_data, color = 'blue', label='Actual Stock Price')
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()

mse = mean_squared_error(test_data, fc)
mae = mean_absolute_error(test_data, fc)
rmse = math.sqrt(mean_squared_error(test_data, fc))
mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
print('MSE: ', str(mse))
print('MAE: ', str(mae))
print('RMSE: ', str(rmse))
print('MAPE: ', str(mape))

training_data = train_data.values
test_data = test_data.values
history = [x for x in training_data]
model_predictions = []
N_test_observations = len(test_data)
for time_point in range(N_test_observations):
    model = ARIMA(history, order=(0,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    true_test_value = test_data[time_point]
    history.append(true_test_value)
MSE_error = mean_squared_error(test_data, model_predictions)
print('Testing Mean Squared Error is {}'.format(MSE_error))

range_s = df[int(len(df)*0.9):].index
plt.plot(range_s, model_predictions, color='blue', marker='o', linestyle='dashed',label='Predicted Price')
plt.plot(range_s, test_data, color='red', label='Actual Price')
plt.title('Stock Prices Prediction')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.legend()
plt.show()
