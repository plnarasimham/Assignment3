#!/usr/bin/env python
# coding: utf-8

# In[1]:

from __future__ import print_function
import sys
from operator import add
from pyspark import SparkContext
import numpy as np
sc = SparkContext()


# In[3]:




def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False

def correctRows(p):
    if(len(p) == 17):
        if(isfloat(p[5]) and isfloat(p[11])):
            if(float(p[5])!=0 and float(p[11])!=0):
                return p

def removeOutlier(p):
    if(float(p[16])>=1 or float(p[16])<=600):
        return p
            


#print("SparkContext is:",sc)
lines = sc.textFile(sys.argv[1])
lines.first()
taxilines = lines.map(lambda x: x.split(','))

taxilinesCorrected = taxilines.filter(correctRows)

#correct the RDD data as per the provided code
taxilines = lines.map(lambda x: x.split(','))    
taxilinesCorrected = taxilines.filter(correctRows)


# In[8]:


taxilinesCorrected.count()


# In[4]:


taxilinesCorrected = taxilinesCorrected.filter(removeOutlier)
taxilinesCorrected.count()


# In[5]:


taxidata = taxilinesCorrected.map(lambda x: (float(x[11]),float(x[5])))
beta = 0.1
num_iteration = 20
learningRate=0.0000001
beta = np.zeros(2)
                                  


# In[6]:


for i in range(num_iteration):
    size = 100
    sample = taxidata.sample(False, size)
    
    
    gradientCost=sample.map(lambda x: (x[1], (x[0] - np.dot(x[1] , beta) )))                           .map(lambda x: (x[0]*x[1], x[1]**2 )).reduce(lambda x, y: (x[0] +y[0], x[1]+y[1] ))
    
    cost= gradientCost[1]
    
    gradient=(-1/float(size))* gradientCost[0]
    
    print(i, "Beta", beta, " Cost", cost)
    beta = beta - learningRate * gradient


# In[ ]:




