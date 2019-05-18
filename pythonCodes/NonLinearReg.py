# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 11:55:49 2018

@author: hejia
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

df=pd.read_csv('china_gdp.csv')
plt.plot(df['Year'],df['Value'])
plt.xlabel('Years')
plt.ylabel('GDP')

x=df['Year'].values
y=df['Value'].values

# choose a function and get the parameters
# in this case, logistic funcion is a good choice, but nomalization is required
def f(x,b1,b2):
    y=1/(1+np.exp(-b1*(x-b2)))
    return y
x_norm=x/max(x)
y_norm=y/max(y)
popt,pcov=curve_fit(f,x_norm,y_norm)
print('the parameters are: ',popt)

# plot the result
plt.figure(figsize=(12,7))
plt.scatter(x_norm,y_norm,label='actual',color='b')
y_p=f(x_norm,*popt)
plt.plot(x_norm,y_p,label='fit',color='r')
plt.legend(loc='best')
plt.xlabel('Year')
plt.ylabel('GDP')
