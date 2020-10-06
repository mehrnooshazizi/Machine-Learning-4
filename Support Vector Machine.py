# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:10:43 2020

@author: Azizi
"""
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('vehicle_csv.csv')
print('\n________ Dataset _________')
print(dataset)

x= dataset[['COMPACTNESS','CIRCULARITY','DISTANCE_CIRCULARITY','RADIUS_RATIO',
            'PR.AXIS_ASPECT_RATIO','MAX.LENGTH_ASPECT_RATIO','SCATTER_RATIO',
            'ELONGATEDNESS','PR.AXIS_RECTANGULARITY','MAX.LENGTH_RECTANGULARITY',
            'SCALED_VARIANCE_MAJOR','SCALED_VARIANCE_MINOR','SCALED_RADIUS_OF_GYRATION',
            'SKEWNESS_ABOUT_MAJOR','SKEWNESS_ABOUT_MINOR','KURTOSIS_ABOUT_MAJOR',
            'KURTOSIS_ABOUT_MINOR','HOLLOWS_RATIO']]
y= dataset['Class']
xtrain, xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3)
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(xtrain, ytrain)
ypredict= classifier.predict(xtest)
ytrainpredict= classifier.predict(xtrain)
accuracytrain= metrics.accuracy_score(ytrain, ytrainpredict)
accuracytest= metrics.accuracy_score(ytest, ypredict)
print('_____________________________Train ACCURACY____________________________')
print(f'Your SVC Accuracy is: {accuracytrain}')
print('_____________________________Test ACCURACY____________________________')
print(f'Your SVC Accuracy is: {accuracytest}')
print('_________________PREDICTION BASED ON YOUR INPUTS_________________')
print(classifier.predict([[109,55,102,169,51,6,241,27,26,165,265,870,247,84,10,11,184,183]]))

