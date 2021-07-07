
from firebase import firebase
from flask import json
from pandas import concat,DataFrame
import numpy as np 
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import webbrowser
from math import sqrt


# In[42]:


data = pd.read_csv("city_hour.csv") 
data=data[data['City']=='Chennai']
data.drop(columns=['City','NOx','NO','O3','AQI_Bucket','Benzene','Toluene','Xylene'],axis=1,inplace=True)
data.interpolate(limit_direction="both",inplace=True)
data.head(5)


# In[43]:


firebase = firebase.FirebaseApplication("https://smartairpollutionmonitoring-default-rtdb.firebaseio.com/")
result = firebase.get('/ifpdata',None)
df = pd.DataFrame.from_dict({(i): result[i] 
                           for i in result.keys()},
                       orient='index')


# In[44]:


df.drop(columns=['Hour','Humidity','Temperature'],axis=1,inplace=True)
dropnan=True
if dropnan:
	df.dropna(inplace=True)
df.interpolate(limit_direction="both",inplace=True)
df.head(5)


# In[45]:


aqi_list=data['AQI'].to_list()
aqi_list_test=df['aqi'].to_list()


# In[46]:


def series_to_supervised(data, n_in, n_out):
	n_vars = 1 if type(data) is list else data.shape[1]
	data = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(data.shift(i))
		names += [('AQI(t-%d)' % (i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(1, n_out):
		cols.append(data.shift(-i))
		names += [('AQI(t+%d)' % (i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	return agg

data1= series_to_supervised(aqi_list, 4, 6)
#print(data1)
data1_test= series_to_supervised(aqi_list_test, 4, 6)
#print(data1_test.tail(5))


# In[47]:


data2 = pd.concat([data.reset_index(drop=True),data1.reset_index(drop=True)], axis=1)
#print(data2)
data2_test = pd.concat([df.reset_index(drop=True),data1_test.reset_index(drop=True)], axis=1)
#print(data2_test)
actual_values=aqi_list_test[-1:]
#print(actual_values)


# In[48]:


#Processing data 

dropnan=True
if dropnan:
	data2.dropna(inplace=True)
dropnan=True
if dropnan:
	data2_test.dropna(inplace=True)


# In[50]:


#Training and the testing set

X_train=np.array(data2[['PM2.5','NO2','NH3','CO','SO2','AQI','AQI(t-4)','AQI(t-3)','AQI(t-2)','AQI(t-1)']])
y_train=data2[['AQI(t+1)']]

X_test=np.array(data2_test[['Ptwo','Nitrates','Ammonia','Carmono','Sulphur','aqi','AQI(t-4)','AQI(t-3)','AQI(t-2)','AQI(t-1)']])
y_test=data2_test[['AQI(t+1)']]


# In[51]:


#Linear regression model

regression_model = linear_model.LinearRegression()
regression_model.fit(X_train,y_train)

#Prediction 

rows,cols=X_test.shape
y_pred=regression_model.predict(X_test)
pred=y_pred[rows-1]


# In[52]:


def linear_reg_predict():
    pred=y_pred[rows-1]
    accuracy = regression_model.score(X_train,y_train) * 100
    rmse = sqrt(mean_squared_error(actual_values,pred))
    return actual_values[0],pred[0],accuracy,rmse


# In[53]:


actual,predicted,acc,rmse = linear_reg_predict()
#Predicted values,Accuracy,and rmse
print("The Actual AQI : ",actual)
print("The predicted AQI : ",predicted)
print("The accuracy of the prediction: ",acc)
print("RMSE: ",rmse)


# In[54]:


#RandomForest Regressor

rf=MultiOutputRegressor(RandomForestRegressor(n_estimators=10,max_depth=20)).fit(X_train, y_train)
y_pred1=rf.predict(X_test)
pred1=y_pred1[rows-1]

def Random_Forest_predict():
    pred1=y_pred1[rows-1]
    accuracy = rf.score(X_train,y_train) * 100
    rmse = sqrt(mean_squared_error(actual_values,pred))
    return actual_values[0],pred[0],accuracy,rmse


# In[55]:


actual1,predicted1,acc1,rmse1 = Random_Forest_predict()
#Predicted values,Accuracy,and rmse
print("The Actual AQI : ",actual1)
print("The predicted AQI : ",predicted1)
print("The accuracy of the prediction: ",acc1)
print("RMSE: ",rmse1)


# In[56]:

data_output = {}
def find_large():

    if(acc>acc1):
        data_output.update({"actual_aqi": actual, "predicted_aqi": predicted, "accuracy": acc, "rmse": rmse})
    else:
        data_output.update({"actual_aqi": actual1, "predicted_aqi": predicted1, "accuracy": acc1, "rmse": rmse1})
    return data_output
find_large()