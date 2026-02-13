# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. .Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Yogesh.S
RegisterNumber:  212224230311
*/

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

data = pd.read_csv("Employee (1).csv")

print("OUTPUT:\nDATA HEAD:")
print(data.head(), "\n")

print("DATASET INFO:")
print(data.info(), "\n")

print("NULL DATASET:")
print(data.isnull().sum(), "\n")

print("VALUES COUNT IN THE LEFT COLUMN:")
print(data['left'].value_counts(), "\n")

le = LabelEncoder()
data['salary'] = le.fit_transform(data['salary'])

print("DATASET TRANSFORMED HEAD:")
print(data.head(), "\n")

X = data[['satisfaction_level', 'last_evaluation', 'number_project',
          'average_montly_hours', 'time_spend_company',
          'Work_accident', 'promotion_last_5years', 'salary']]
y = data['left']

print("X.HEAD:")
print(X.head(), "\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("ACCURACY:", accuracy, "\n")

sample = [[0.5, 0.8, 9, 260, 6, 0, 1, 2]]
prediction = dt.predict(sample)
print("DATA PREDICTION:", prediction, "\n")
```

## Output:

### DATA HEAD:

<img width="745" height="394" alt="image" src="https://github.com/user-attachments/assets/057c4bd7-11f3-4728-a2f8-469edec4a5be" />

### DATASET INFO:

<img width="389" height="255" alt="image" src="https://github.com/user-attachments/assets/69720654-49eb-4c01-ada1-a362703bc9dc" />

### NULL DATASET:

<img width="472" height="108" alt="image" src="https://github.com/user-attachments/assets/4cf5120a-8c2d-4212-b03d-7ade9fd6c401" />


### VALUES COUNT IN THE LEFT COLUMN:

<img width="933" height="172" alt="image" src="https://github.com/user-attachments/assets/7b6b0a45-5cd4-4fd4-8645-aadb0ebed423" />


### X.HEAD:

<img width="287" height="60" alt="image" src="https://github.com/user-attachments/assets/1b94ea88-21c5-4487-91ac-d919ed1cd6a3" />


### ACCURACY:

<img width="340" height="57" alt="image" src="https://github.com/user-attachments/assets/5e3e6e56-8ebd-4fc5-87cf-58187ba6ab3f" />


### DATA PREDICTION:

<img width="226" height="51" alt="image" src="https://github.com/user-attachments/assets/ad9adb66-7a58-42d6-aba6-6b1d106fcd2b" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
