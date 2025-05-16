# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries for data handling, preprocessing, modeling, and evaluation.
2.Load the dataset from the CSV file into a pandas DataFrame.
3.Check for null values and inspect data structure using .info() and .isnull().sum().
4.Encode the categorical "Position" column using LabelEncoder.
5.Split features (Position, Level) and target (Salary), then divide into training and test sets.
6.Train a DecisionTreeRegressor model on the training data.
7.Predict on test data, calculate mean squared error and R² score, and make a sample prediction.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Rishab p doshi
RegisterNumber:  212224240134
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error
data=pd.read_csv("/content/Salary.csv")
print(data.head())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
print(data.head())

x=data[["Position","Level"]]
y=data["Salary"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("accuracy:",accuracy_score(y_test,y_pred))
print("mean squared error:",mean_squared_error(y_test,y_pred))
sc=r2_score(y_pred,y_test)
print("r2: ",sc)
print(model.predict([[5,6]]))
```
## Output:
![image](https://github.com/user-attachments/assets/fc19dce8-7ffc-41b5-8819-b1c8edb5d5db)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
