# -*- coding: utf-8 -*-

"""
@author: Bahar Dorri

"""
import numpy as np
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot as plt

def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
   # print(columns)

    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    #print(df)
    return df

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return history-yhat 

 # scale train and test data to [-1, 1]
def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1]) #matrix format
    train_scaled = scaler.transform(train)
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

def invert_scale(scaler, X, value):
    #print('scaler:',scaler,'X:',X,'value:',value)
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]
 
def fit_lstm(train, batch_size,nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    #The LSTM layer expects input to be in a matrix with the dimensions: [samples, time steps, features].
    print('X',X.shape)
    print('y',y.shape)
    X = X.reshape(X.shape[0], 1, X.shape[1])    
    model = Sequential() #make linear stack of layers
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=False))#, return_sequences=True))
    model.add(Dropout(0.2))  
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):    
        model.fit(X, y, epochs=1, batch_size=batch_size, shuffle=True, verbose=0)
       # model.reset_states()   
    return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
   # print('X1',X)
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    #print('yhat-forecast',yhat,'X2',X)
    return yhat[0,0]

series = read_csv('mindenSum.csv')
totalData=series['L3']
raw_Part = totalData
differenced = difference(raw_Part, 1)
supervised = timeseries_to_supervised(differenced, 1)
supervised_values = supervised.values
print(supervised_values)
raw_test=[]
m=100#1000
#t=1800
n=130#130#1300
#train, test = supervised_values[0:m], supervised_values[m:n]
train = supervised_values[0:m]


series2 = read_csv('Daily_BRG_Mid_City_Levels.csv')
totalData2=series2['L3']
raw_Part2 = totalData2
differenced2 = difference(raw_Part2, 1)
supervised2 = timeseries_to_supervised(differenced2, 1)
supervised_values2 = supervised2.values
test= supervised_values2[0:30]

scaler, train_scaled, test_scaled = scale(train, test)
print('train_scaled',train_scaled.shape,', test_scaled', test_scaled.shape)
TrainLen=len(train_scaled)
TestLen=len(test_scaled)
a=raw_Part2[0:30]
j=m
b=raw_Part[0:m]
raw_B=[]
for i in range (TrainLen):
    raw_B.append(b[i])
for i in range (TestLen):
    raw_test.append(a[i])
    #raw_test.append(a[j+i])

batch_size=1 #This is because it must be a factor of the size of the training and test datasets.
repeats=1
error_scores=list()
predB=list()

for r in range(repeats):
    lstm_model = fit_lstm(train_scaled, batch_size,1000,4 )
# forecast the entire training dataset to build up state for forecasting
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)#requires a 3D NumPy array as input 
    B1=lstm_model.predict(train_reshaped, batch_size=batch_size)
    for i in range(TrainLen):
        XB,yB = train_scaled[i, 0:-1],train_scaled[i,-1]
        B2 = invert_scale(scaler, XB, B1[0,0])
        #print('B2:',B2,'XB',XB)
        B2 = inverse_difference(raw_B[-(TrainLen-i)], B2)#, len(test_scaled)-i)
        #print('TrainLen-i:',TrainLen-i,'raw_B[-(TrainLen-i)]:',raw_B[-(TrainLen-i)],'B2:',B2)
        predB.append(B2)
        expectB = raw_B[i]
       # print('Predicted=',round(B2,0), ', Expected=', expectB)
            
    predictions = list()
    for i in range(TestLen):
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        
        yhat = forecast_lstm(lstm_model, batch_size, X)
        #print('yhat',yhat)
        yhat = invert_scale(scaler, X, yhat)
       # print('yhat-invertScale',yhat)
        
        yhat = inverse_difference(raw_test[-(TestLen-i)], yhat, len(test_scaled)-i)
        if yhat<0 :
            yhat=0
        predictions.append(yhat)
        expected = raw_test[i]
        print('Predicted=',round(yhat,0), ', Expected=', expected)
        
    
    rmse = sqrt(mean_squared_error(raw_test, predictions))
    print('%d) Test RMSE: %.3f' % (r+1, rmse))
    error_scores.append(rmse)

plt.plot(raw_test, 'orange', label='expected')
plt.plot(predictions,'blue', label='predictions')
plt.title('predictions and expected')
plt.legend()
plt.show()