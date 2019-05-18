# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 18:43:08 2018

@author: hejia
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

df=pd.read_csv('Fuel.csv')
df.head()

df.describe()# count, mean,min,25%,50%,75%
values = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

# plot the relationship between engine size and co2 emission
plt.figure(figsize=(12,7))
plt.scatter(values['ENGINESIZE'],values['CO2EMISSIONS'])
plt.xlabel('engine size')
plt.ylabel('co2')

#plot the relatiohsip between cylinder and co2 emission
plt.figure(figsize=(12,7))
plt.scatter(values['CYLINDERS'],values['CO2EMISSIONS'])
plt.xlabel('cylinder size')
plt.ylabel('co2')

# get train and test data, train: 80%, test: 20%
msk=np.random.rand(len(values))<0.8
train=values[msk]
test=values[~msk]# ~ works for array not list

# linear regression model
linear=linear_model.LinearRegression()
train_x=np.array(train['ENGINESIZE']).reshape(-1,1)
train_y=np.array(train['CO2EMISSIONS']).reshape(-1,1)
linear.fit(train_x,train_y) # fit(x,y)
print('the coef is {0}, the intercept is {1}'.format(linear.coef_,linear.intercept_))

# plot predicted data
plt.figure(figsize=(12,7))
plt.scatter(train['ENGINESIZE'],train['CO2EMISSIONS'])
plt.plot(train_x,linear.coef_[0][0]*train_x + linear.intercept_[0],color='r')
plt.xlabel('engine')
plt.ylabel('Emission')

# test the value

test_x=np.array(test['ENGINESIZE']).reshape(-1,1)
test_y=np.array(test['CO2EMISSIONS']).reshape(-1,1)
predict_y=linear.predict(test_x)

print('The mean of square error is {}'.format(np.mean((predict_y-test_y)**2)))
print('R2 is {}'.format(r2_score(test_y,predict_y))) # R2 the bigger the better



