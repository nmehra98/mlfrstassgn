# IMPORTING ALL REQUIRED LIBRARIES
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import*
import os
from sklearn import preprocessing
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# LOADING THE DATASET : FASHION MNIST
df = pd.read_csv("C:\\Users\\dell\\Desktop\\MACHINE_LEARNING\\fashion-mnist_train.csv")
print(df)

# X CONTAINS DATA EXCEPT COLUMNN - "LABEL"
# Y CONTAINS COLUMN - "LABEL"
x=df.drop(["label"],axis=1)
y=df["label"]
x.shape,y.shape
x.describe

# NORMALIZING X
z = preprocessing.normalize(x)
z
z1 = preprocessing.scale(z)
z1
z1.shape
# NOW z1 IS NEW x

# VIEWING THE IMAGE 
plt.imshow(z1[0].reshape(28,28))

# IMPLEMENTING CLASSIFIER MODELS
# BAGGING CLASSIFIER
model = DecisionTreeClassifier()
num_trees = 100
model1 = BaggingClassifier(base_estimator=model)
model1

# SPLITTING THE DATA INTO TRAIN AND TEST
z1_train,z1_test,y_train,y_test=train_test_split(z1,y,test_size=0.3)

model1.fit(z1_train,y_train)
pred = model1.predict(z1_test)
metrics.accuracy_score(y_test,pred)
print(classification_report(y_test,pred))
confusion_matrix(y_test,pred)

# RANDOM FOREST CLASSIFIER
rf=RandomForestClassifier()
rf.fit(z1_train,y_train)
pred1 = rf.predict(z1_test)
metrics.accuracy_score(y_test,pred1)
print(classification_report(y_test,pred1))
confusion_matrix(y_test,pred1)

# GRADIENT BOOSTING CLASSIFIER
model2 = GradientBoostingClassifier(n_estimators=30,verbose=1)
model2
model2.fit(z1_train,y_train)
pred2 = model2.predict(z1_test)
metrics.accuracy_score(y_test,pred2)
print(classification_report(y_test,pred2))
confusion_matrix(y_test,pred2)

# COMPARING DIFFERENT CLASSIFIERS
list = []
h1 = LogisticRegression()
h2 = DecisionTreeClassifier()
h3 = svm.SVC()
list = VotingClassifier([('LogisticRegression', h1), ('DecisionTreeClassifier', h2), ('SVM', h3)])
list
list.fit(z1_train,y_train)
pred3 = list.predict(z1_test) 
metrics.accuracy_score(y_test,pred3)
print(classification_report(y_test,pred3))
confusion_matrix(y_test,pred3)


# EVERY MODEL'S ACCURACY SCORE, CLASSIFICATION REPORT AND CONFUSION MATRIX IS ALSO SHOWN...!