import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing,neighbors
from sklearn.model_selection import train_test_split

ds = pd.read_csv(r'//Users/rahulmanoj/Desktop/Python /water_potability.csv')
ds = ds.fillna(0)
# print(ds.head())
X = np.array(ds.drop(['Potability'],1))
X = pd.DataFrame(X)
# print(X.head())
y = np.array(ds['Potability'])
y = pd.DataFrame(y)
# print(y.head())
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)
