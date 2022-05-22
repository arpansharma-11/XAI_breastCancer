# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:07:37 2022

@author: Arpan
"""

from data_setup import df
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set up the data for modelling 
# define Y 
y=df['target'].to_frame() 
# define X df.columns.difference removes the specified column
X=df[df.columns.difference(['target'])] 
 # create train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


# build model - Xgboost

# build classifier
xgb_mod=xgb.XGBClassifier(random_state=101, gpu_id=0, use_label_encoder = False)
#values.ravel() compresses any array to 1D Array 
#print(y_train.values.ravel())
xgb_mod=xgb_mod.fit(X_train, y_train.values.ravel())   

#print(xgb_mod)


# make prediction and check model accuracy 
y_pred = xgb_mod.predict(X_test)

#print(y_pred)

#probable question : What if we change the dataset to Influenza Dataset?

# Performance
#Actual Prediction Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))