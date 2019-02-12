#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:36:56 2019

@author: chandrakantachaudhury
"""

import pandas as pd
import os
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.grid.grid_search import H2OGridSearch

from h2o.estimators.xgboost import H2OXGBoostEstimator
from __future__ import print_function
h2o.init()
#change directory 
os.chdir("/home/chandrakantachaudhury/Desktop/HACKTHON/thinkdata")

#read csv file ->top_5crime , after EDA file has been saved as csv 
df=pd.read_csv("top_5crime.csv")

df.head(3) 

#delete year before 2008, considered crime records from 2008 to 2018
df=df[df['Year'] > 2007]

#converting from pandas dataframe to h2o frame for h2o ml feasibility
h2oframe = h2o.H2OFrame(df)
#deleted pandas dataframe to release space
del df

#split dataset to train , test ,validaiton (60 %, 20% ,20% )
train,test,valid = h2oframe.split_frame(ratios=[0.6,0.2])

#XGB model  training
xgbmodel = H2OXGBoostEstimator(keep_cross_validation_fold_assignment = True,nfolds =10,seed=1234,keep_cross_validation_predictions=True)
xgbmodel.train(y="Primary Type", x=["Block", "IUCR", "Location Description", "Arrest","Domestic","Beat",
                "District","FBI Code","Year","Latitude","Longitude","month","weekday","hour"],training_frame = train, validation_frame = valid)

#random forest model training
rfmodel = H2ORandomForestEstimator(keep_cross_validation_fold_assignment = True,nfolds =10,seed=1234,keep_cross_validation_predictions=True)
rfmodel.train(y="Primary Type", x=["Block", "IUCR", "Location Description", "Arrest","Domestic","Beat",
                "District","FBI Code","Year","Latitude","Longitude","month","weekday","hour"],training_frame = train, validation_frame = valid)

# ensemble both XGB and random forest 
ensemble = H2OStackedEnsembleEstimator(model_id="chicagocrime_ensemble",
                                       base_models=[xgbmodel,rfmodel])

#train ensemble model on train data part
ensemble.train(y="Primary Type", x=["Block", "IUCR", "Location Description", "Arrest","Domestic","Beat",
                                      "District","FBI Code","Year","Latitude","Longitude","month","weekday","hour"],
                                         training_frame=train,validation_frame = valid)

#sav the model
model_path = h2o.save_model(model=ensemble, path ="/home/chandrakantachaudhury/Desktop/HACKTHON/thinkdata", force=True)
model_path="/home/chandrakantachaudhury/Desktop/HACKTHON/thinkdata/chicagocrime_ensemble"

#load saved ensemble model
saved_model = h2o.load_model(model_path)


#predict the test data 
 
predicttest=saved_model.predict(test) 

#converting  predicted value to pandas dataframe
pred_test=predicttest.as_data_frame()

#conveting test data to pandas dataframe
actual_test=test.as_data_frame()

#copy only  target value  from test data and  predicted data to separate series ,which can be compared later for accuracy
testframe=actual_test["Primary Type"].copy()

predframe=pred_test["predict"].copy()


#plot confusion matrics  

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(testframe, predframe)

print(cm)


# check for accuracy 

from sklearn.metrics import accuracy_score
acuracy=accuracy_score(testframe, predframe)

print(acuracy)



ensemble.model_performance(test) 