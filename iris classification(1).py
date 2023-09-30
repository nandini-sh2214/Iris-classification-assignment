

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('Iris.csv')

data.columns
data.isna().sum()
data=pd.read_csv('Iris.csv',na_values=[0])

columns=['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

for column in columns:
    mean=data[column].mean()
    data[column]=data[column].replace(0,mean)
    
    
#############################
#data['Id'].mean()
#data['Id'].replace(0,mean)
##############################################
plt.figure(figsize=(12,8))
sns.pairplot(data)

data.columns
columns=['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

for column in columns:
    plt.figure(figsize=(10,8))
    sns.boxplot(x=data["Species"],y=data[column])
    print('\n')
    
###########################################
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True)
##################################
x=data.drop(['Species'],axis=1)
y=data['Species']
x=data.drop(['Species'],axis=1)
y=data['Species']
##############################################
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,
                            test_size=0.20,
                            random_state=0)
##################################
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
columns=['Id','SepalLengthCm','SepalWidthCm']
for column in columns:
    data[column]=encoder.fit_transform(data[column])
#################################

x=data.drop(['Species'],axis=1)
y=data['Species']
#################################


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
#################################

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(x_train,y_train)
#################################

regressor.coef_ 
regressor.intercept_ 
#################################

y_pred=regressor.predict(x_test)

from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_test, y_pred)) 
metrics.mean_absolute_error(y_test, y_pred)
metrics.r2_score(y_test, y_pred) 
