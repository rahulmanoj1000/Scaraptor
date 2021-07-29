import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

ds = pd.read_csv(r'//Users/rahulmanoj/Desktop/Python /Salary_Data.csv')
X = ds[['YearsExperience']]
y = ds[['Salary']]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,)
clf = LinearRegression()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
New_data = [[1],[34],[45]]
clf.predict(New_data)
pred = clf.predict(New_data)
print(pred)
