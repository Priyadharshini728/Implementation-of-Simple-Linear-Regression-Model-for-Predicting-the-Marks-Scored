# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

*/
```
## Developed by: PRIYADHARSHINI P
## RegisterNumber: 212224040252

## Output:

Head values:

![image](https://github.com/user-attachments/assets/a99f8742-40c0-4b54-b563-ce4347a20103)

Tail values:

![image](https://github.com/user-attachments/assets/1094bfa9-467a-433e-a643-ed382ac1ac71)

Array value of X:

![image](https://github.com/user-attachments/assets/63c82cd8-bd30-49fd-a976-70b88306d603)

Array value of Y:

![image](https://github.com/user-attachments/assets/bcb1e3d5-8ff6-45d9-a56d-347f766cd22f)

Values of Y prediction:

![image](https://github.com/user-attachments/assets/200eb18f-4a9f-4e5e-aade-16dc6b57b294)

Array values of Y test:

![image](https://github.com/user-attachments/assets/4306bdda-4675-4576-82d3-4c8fea49ba5e)

Training set graph:

![image](https://github.com/user-attachments/assets/f568cca8-ee9d-4642-bb8e-08b0e3e65912)

Test set graph:

![image](https://github.com/user-attachments/assets/1a2812be-beae-45cf-b3d3-d475fc1957f0)

Values of MSE, MAE and RMSE:

![image](https://github.com/user-attachments/assets/7ef7746a-f3d8-41d4-9710-5ebb54b4d333)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
