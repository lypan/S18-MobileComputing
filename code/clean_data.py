#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
#%% sensor
inputFileName = '../data/sensor_data.csv'
outputFileName = '../data/clean_sensor_data.csv'
#%%
df = pd.read_csv(inputFileName, names=["time", "latitude", "longtitude", "pm"])
df = df[df.pm != -1.0]
df = df[df.pm < 500.4]
df.time = df.time % (24 * 7)
df = df.sort_values(['time', "latitude", "longtitude",])
df.to_csv(outputFileName, encoding='utf-8', index=False, header=False)
#%% 
inputFileName = '../data/phy_data.csv'
outputFileName = '../data/clean_phy_data.csv'
#%%
df = pd.read_csv(inputFileName, names=["time", "latitude", "longtitude", "pm"])
df.time = df.time % (24 * 7)
df = df[df.pm < 500.4]
df.to_csv(outputFileName, encoding='utf-8', index=False, header=False)
