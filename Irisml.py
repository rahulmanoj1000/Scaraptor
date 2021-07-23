import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn .linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



df = pd.read_csv(r'//Users/rahulmanoj/Desktop/Python /iris2.csv')
df = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']]
# print(df.head())
x = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
# x = StandardScaler().fit_transform(x)
y = df[['Species']]
x_train,x_test, y_train,y_test = train_test_split(x,y,test_size = 0.05,random_state = False)
clf = LinearRegression()
clf.fit(x_train,y_train)
accuracy = clf.score(x_test,y_test)
print(accuracy)
print()
clf.predict(x_test)
pred = clf.predict(x_test)
print(pred)
print(x_test)
