## DATE: 26.04.2024
## EXPERIMENT: 09
# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## Algorithm
1. Start the program
2. Import the python pandas library as pd
3. Read the contents of the Spam csv file
4. Display the first 5 rows of the dataset using head()
5. Assign x as v1 values and y as v2 values
6. From sklearn library select the feature extraction and import CountVectorizer
7. CountVectorizer will convert the Text to Numerical Data
8. From sklearn library import Support Vector Classifier (ie. SVC)
9. Predict the x_test using SVC
10. Print the accuracy of the SVM Model
11. Stop the program


## Program:
```PYTHON
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Jeevanesh.S
RegisterNumber:  212222243002
*/


import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='Windows - 1252')

data.head()

data.info()

data.isnull().sum()

x=data['v1'].values
y=data['v2'].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

### RESULT OUTPUT

![image](https://github.com/Sachin-vlr/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497666/7fc28e96-c69c-425f-aa26-bbd633dd44c0)

### DATA.HEAD()
![image](https://github.com/Sachin-vlr/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497666/747c9703-064a-4fbd-9f09-c685c2339df9)

### DATA.INFO()
![image](https://github.com/Sachin-vlr/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497666/09533dd4-7f5e-4493-9cba-35091a46d7d1)

### DATA.ISNULL().SUM()
![image](https://github.com/Sachin-vlr/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497666/04619be3-255a-40cf-a6a7-5009c4a7820e)

### Y_PREDICTION VAUE
![image](https://github.com/Sachin-vlr/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497666/0bbcc78e-5645-4e4f-91f4-e584ea372e7f)

### ACCURACY VALUE
![image](https://github.com/Sachin-vlr/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497666/bf4518d1-2af2-4b8e-a582-bbd3b4178d41)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
