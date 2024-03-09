# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Santharamanath
RegisterNumber: 212223220097
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:
Data Information
![311409229-e9b1fc7d-16f9-49f4-a3a6-1501751fc2a9](https://github.com/Santharamanath/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/149035289/986b932e-cd55-42eb-9b0d-aa97a2142a42)

Value of X
![311409217-8fe400bf-ef4d-4a65-bbf8-c4c193902a6c](https://github.com/Santharamanath/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/149035289/39df1666-313e-4f6b-b41b-eec83bb26546)

Predicted Value
![311409197-7f5cba3d-289a-4fc6-9779-df0040429087](https://github.com/Santharamanath/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/149035289/b422a607-5211-40f9-a48b-e999c20acaa4)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
