#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from scipy.io import *
import numpy as np
import pandas as pd
#%%
inputFileName = 'data/data_interp_all.mat'
#%%
mat = loadmat(inputFileName)
mdata = mat['data_interp_all']  # variable in mat file
time, latitude, longtitude = mdata.shape

for t in range(time):
    for lat in range(latitude):
        for lon in range(longtitude):
            print("%d,%d,%d,%d" %(t, lat, lon, mdata[t][lat][lon]))