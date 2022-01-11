import compare
import pickle
import pandas as pd
import numpy as np
from pandas import read_csv
from compare import Compare

def PredFromClust(csv,periods):
    data = csv
    diff = Compare(data)
    filename = '{}clustmodel'.format(diff.index[0])
    infile = open(filename,'rb')
    newmod = pickle.load(infile)
    infile.close()
    length = len(data)
    
    prediction = newmod.predict(start=length, n_periods=periods)
    return prediction


