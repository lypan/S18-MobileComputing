#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
#%% sensor
inputFileName = '../data/clean_sensor_data.csv'
#%%
df = pd.read_csv(inputFileName, names=["time", "latitude", "longtitude", "pm"])
data = df[df.columns[:-1]]
label = df["pm"]

#X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, stratify=y)
#%%
bins = np.linspace(0, 500, 500)
label_bin = np.digitize(label, bins)
#%%
skf = StratifiedKFold(shuffle=True, random_state=None, n_splits=2)
for train_index, test_index in skf.split(data, label_bin):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data.iloc[train_index, :], data.iloc[test_index, :]
    y_train, y_test = label_bin[train_index], label_bin[test_index]


lm = LogisticRegression(verbose=True)
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
print ("Score:", model.score(X_test, y_test))

#%% phy
inputFileName = './data/clean_phy_data.csv'
#%%
df = pd.read_csv(inputFileName, names=["time", "latitude", "longtitude", "pm"])
data = df[df.columns[:-1]]
label = df["pm"]

#X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, stratify=y)
#%%
bins = np.linspace(0, 500, 500)
label_bin = np.digitize(label, bins)
#%%
skf = StratifiedKFold(shuffle=True, random_state=None, n_splits=2)
for train_index, test_index in skf.split(data, label_bin):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data.iloc[train_index, :], data.iloc[test_index, :]
    y_train, y_test = label_bin[train_index], label_bin[test_index]


lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
print ("Score:", model.score(X_test, y_test))
    
    