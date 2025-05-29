# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Jayaseelan U
RegisterNumber:  212223220039
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

print(y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

print(x_train.shape)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
print(acc)

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```
## Output:
## x.shape() and y.shape()
![Screenshot 2025-05-23 152604](https://github.com/user-attachments/assets/e21d96a5-f246-4756-8f44-21a264005a4c)
## acc (accuracy)
![Screenshot 2025-05-23 153808](https://github.com/user-attachments/assets/3004b234-0feb-4ff4-994f-76af3f6a5eeb)
## con (confusion matrix)
![Screenshot 2025-05-23 153857](https://github.com/user-attachments/assets/6ce28feb-5f53-4583-a8b7-bc0924afb83c)
## cl (classification report)
![Screenshot 2025-05-23 154000](https://github.com/user-attachments/assets/edef8c6e-f7b4-487b-b53a-5204fe753f7b)
## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
