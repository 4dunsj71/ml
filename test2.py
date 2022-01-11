import compare
import predfromclust
import pred
import pandas as pd
import numpy as np
from pandas import read_csv
from compare import Compare
from predfromclust import PredFromClust
from pred import ArimaPred
data = read_csv('csv/closeaapl.csv')
print(data.shape)
#diff = Compare(data)
#prediction = ArimaPred(data,10)
#print(prediction[prediction.shape[0]-1])
#if diff[1]<20:
#    PredFromClust(data,10)
#else:
    #ArimaPred(data,10)
