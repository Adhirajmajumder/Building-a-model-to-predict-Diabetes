# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:13:43 2020

@author: ADHIRAJ MAJUMDAR
"""

import pandas as pd
import numpy as np
data = pd.read_csv("./dataset/pima-indians-diabetes.data",header=None)
#Use the .NAMES file to view and set the features of the dataset
col = ["pregnant","glucose","bp","skin","insulin","bmi","pedigree","age","label"]
#Use the feature names set earlier and fix it as the column headers of the dataset
data = pd.read_csv("./dataset/pima-indians-diabetes.data",header=None,names=col)
#print(data.head())
feature_object = ["pregnant","glucose","insulin","bmi","age","pedigree"]

#Create the feature object
X_feature = data[feature_object]

#Create the reponse object
Y_target = data["label"]

#View the shape of the feature object
X_feature.shape

#Split the dataset to test and train the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_feature,Y_target,test_size=0.30,random_state=4)


# Create a logistic regression model using the training set
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression()
logReg.fit(X_train,y_train)


#Make predictions using the testing set
y_pred = logReg.predict(X_test) 

#Print the first 30 actual and predicted responses
print('Actual    : ',y_test.values[:30])
print('Predicted : ',y_pred[:30])

#Evaluate the accuracy of your model
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))