# Project Light done by Alireza Rashidi Laleh
# converting a dataframe to actual numpy arrays, ready to be fed into our model
# python 3.7.8


import numpy as np
import pandas as pd


data = pd.read_csv('data.txt', header=None)
print(data.head())
del data[26]

data = data.to_numpy()
print(data)

np.savetxt('preprocessd data.txt', data)
