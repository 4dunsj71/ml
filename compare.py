import pandas as pd
import numpy as np
import csv
from pandas import read_csv
    
def Compare(csv):
    data = csv
    newdata = data['Close']
    diff = pd.DataFrame()
    for x in range(0,4):
        clust = read_csv('csv/{}_average.csv'.format(x))
        clust = clust['Close']
        diff[x] = newdata.subtract(clust)
        

#convert all negative differences to positive values
    negative = diff[diff<0]
    negative = negative * -1
    diff[diff<0] = negative
    #print(diff)    
    minval = diff.min()
    #print(minval)
    minval.sort_values(axis=0,ascending=True,inplace=True)
    
    vals = [minval.index[0],minval[0]]
    valsS = pd.Series(vals)
    return valsS

