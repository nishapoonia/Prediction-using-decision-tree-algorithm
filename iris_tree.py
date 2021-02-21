# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 23:12:07 2021

@author: NISHA POONIA
Prediction using Decision Tree Algorithm
"""

import pandas as pd
import matplotlib.pyplot as plt


#LOADING DATASET
data = pd.read_csv('Iris.csv')
data.drop('Id', inplace=True, axis=1)
print(data)

#Contains all the attributes of the flower
X = data.iloc[:,:-1].values    
print("x:")
print(X)
#Contains the target species of the flowers
Y = data['Species']
print("y:")
print(Y)

data.describe()


#splitting data set in training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)
print("X TRAIN: ",X_train)
print("Y TRAIN: ",Y_train)
print("X TEST: ",X_test)
print("Y TEST: ",Y_test)


#define decision tree
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state = 0)
clf.fit(X_train, Y_train)


#visualizing the decision tree
fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('imagename.png')

#making predictions
y_pred= clf.predict(X_test)
print("predictions:",y_pred)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test,y_pred)
print("ACCURACY OF MODEL:",accuracy)