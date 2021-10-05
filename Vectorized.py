#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import sys
from operator import add
from pyspark import SparkContext
import numpy as np
sc = SparkContext()


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

taxilinesCorrected = taxilinesCorrected.filter(removeOutlier)
taxidata = taxilinesCorrected.map(lambda x: (float(x[11]),float(x[5])))

x_i = taxidata.map(lambda x: x[1])
y_i = taxidata.map(lambda x: x[0])
x_i_y_i = taxidata.map(lambda x: x[0]*x[1])
x_i_2 = taxidata.map(lambda x: x[1]*x[1])

n = taxidata.count()
print("Number of data points = ",n)


x_i_sum = x_i.sum()
y_i_sum = y_i.sum()
x_i_y_i_sum = x_i_y_i.sum()
x_i_2_sum = x_i_2.sum()


m = (n*x_i_y_i_sum - (x_i_sum)*(y_i_sum))/((n*x_i_2_sum)-(x_i_sum)*(x_i_sum))
b = ((x_i_2_sum*y_i_sum) - (x_i_sum*x_i_y_i_sum))/((n*x_i_2_sum)-(x_i_sum)*(x_i_sum))

print("Slope parameter m is:",m)
print("Intercept b is:",b)





