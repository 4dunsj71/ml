import pickle
import datetime as dt
from datetime import date
import pandas as pd
import predtopickle
from predtopickle import ArimaPred
import numpy as np
from pandas import read_csv
def MakePickles():
    
    for x in range(0,4):
        #print(x)
        filename = '{}clustmodel'.format(x)
        print('pickling ',filename)
        data = read_csv('csv/{}_average.csv'.format(x))
        outfile = open(filename,'wb')
       
        model = ArimaPred(data)

        pickle.dump(model,outfile)

        outfile.close()

        



