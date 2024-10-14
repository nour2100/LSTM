#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
#read the data
data = pd.read_csv('MSFT-2.csv')

data


# In[2]:


type(data)


# In[3]:


data.tail()


# In[7]:


opn = data[['Open']]


# In[8]:


#Plot open column information
opn.plot()


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


ds = opn.values
ds


# In[11]:


plt.plot(ds)


# In[12]:


import numpy as np

from sklearn.preprocessing import MinMaxScaler


# In[13]:


#Using MinMaxScaler for normalizing data between 0 & 1
normalizer = MinMaxScaler(feature_range=(0,1))
ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))

len(ds_scaled), len(ds)


# In[43]:


ds_scaled


# In[14]:


#Defining test and train data sizes
train_size = int(len(ds_scaled)*0.70)
test_size = len(ds_scaled) - train_size

train_size,test_size


# In[15]:


#Splitting data between train and test
ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1]


# In[16]:


#creating dataset in time series for LSTM model 
#X[100,120,140,160,180] : Y[200]
def create_ds(dataset,step):
    Xtrain, Ytrain = [], []
    for i in range(len(dataset)-step-1):
        a = dataset[i:(i+step), 0]
        Xtrain.append(a)
        Ytrain.append(dataset[i + step, 0])
    return np.array(Xtrain), np.array(Ytrain)


# In[17]:


#Taking 100 days price as one record for training
time_stamp = 100
X_train, y_train = create_ds(ds_train,time_stamp)
X_test, y_test = create_ds(ds_test,time_stamp)


# In[18]:


#Reshaping data to fit into LSTM model
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[19]:


from keras.models import Sequential
from keras.layers import Dense, LSTM

#Creating LSTM model using keras
model = Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
model.add(LSTM(units=50,return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1,activation='relu'))
model.summary()


# In[20]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
#Training model with adam optimizer and mean squared error loss function
model.compile(loss='mse', 
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64)


# In[21]:


#PLotting loss, it shows that loss has decreased significantly and model trained well
loss = model.history.history['loss']
plt.plot(loss)


# In[23]:


hist = pd.DataFrame(model.history.history)
hist.head()


# In[25]:


#Plot history 
hist.plot()


# In[26]:


#Predicitng on train and test data
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# In[27]:



#Inverse transform to get actual value
train_predict = normalizer.inverse_transform(train_predict)
test_predict = normalizer.inverse_transform(test_predict)

#Comparing using visuals
plt.plot(normalizer.inverse_transform(ds_scaled))
plt.plot(train_predict)
plt.plot(test_predict)


# In[28]:


type(train_predict)

test = np.vstack((train_predict,test_predict))

#Combining the predited data to create uniform data visualization
plt.plot(normalizer.inverse_transform(ds_scaled))
plt.plot(test)


# In[29]:


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[30]:



### Test Data RMSE
math.sqrt(mean_squared_error(y_test,test_predict))


# In[31]:


### Plotting 
# shift train predictions for plotting
train_predict = normalizer.inverse_transform(train_predict)
test_predict = normalizer.inverse_transform(test_predict)
look_back=100
trainPredictPlot = np.empty_like(ds)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(ds)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(ds)-1, :] = test_predict
# plot baseline and predictions
plt.plot(normalizer.inverse_transform(ds))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[32]:


len(ds_test)


# In[33]:


#Getting the last 100 days records
fut_inp = ds_test[353:]


# In[34]:






fut_inp = fut_inp.reshape(1,-1)

tmp_inp = list(fut_inp)

fut_inp.shape


# In[35]:


#Creating list of the last 100 data
tmp_inp = tmp_inp[0].tolist()


# In[36]:


#Predicting next 30 days price suing the current data
#It will predict in sliding window manner (algorithm) with stride 1
lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(tmp_inp)>100):
        fut_inp = np.array(tmp_inp[1:])
        fut_inp=fut_inp.reshape(1,-1)
        fut_inp = fut_inp.reshape((1, n_steps, 1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        tmp_inp = tmp_inp[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        fut_inp = fut_inp.reshape((1, n_steps,1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[44]:


len(lst_output)


# In[38]:


#Creating a dummy plane to plot graph one after another
plot_new=np.arange(1,101)
plot_pred=np.arange(101,131)

plt.plot(plot_new, normalizer.inverse_transform(ds_scaled[1410:]))
plt.plot(plot_pred, normalizer.inverse_transform(lst_output))


# In[39]:


ds_new = ds_scaled.tolist()

len(ds_new)


# In[40]:


#Entends helps us to fill the missing value with approx value
ds_new.extend(lst_output)
plt.plot(ds_new[1200:])


# In[41]:


#Creating final data for plotting
final_graph = normalizer.inverse_transform(ds_new).tolist()


# In[42]:




#Plotting final results with predicted value after 30 Days
plt.plot(final_graph,)
plt.ylabel("Price")
plt.xlabel("Time")
plt.title("{0} prediction of next month open")
plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'NEXT 30D: {0}'.format(round(float(*final_graph[len(final_graph)-1]),2)))
plt.legend()


# In[ ]:




