#!/usr/bin/env python
# coding: utf-8
##" Application of Linear Regression on Swedish Auto Insurance Dataset
#The Swedish Auto Insurance Dataset involves predicting the total payment for all claims in thousands of Swedish Kronor, given the total number of claims.

#It is a regression problem. It is comprised of 63 observations with 1 input variable and one output variable. The variable names are as follows:

#Number of claims.
#Total payment for all claims in thousands of Swedish Kronor"# Standard Library Includes
We will be importing the below libraries

numpy - NumPy is the fundamental package for scientific computing with Python
pandas - Pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools
matplotlib - 2D plotting library which produces publication quality figures in a variety of formats and interactive environments across platforms
seaborn - Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics
sklearn - Library that implements a range of machine learning, preprocessing, cross-validation and visualization algorithms
# In[91]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[92]:


data=pd.read_csv("auto_insurance_sweden.csv")


# In[93]:


data.head()


# In[94]:


#Declaring the dependent and independent variable 


# In[95]:


X = data.iloc[:, :-1].values
Y = data.iloc[:, 1].values


# In[96]:


# Splitting the dataset into the Training set and Test set

Visualization using seaborn and matplotlib
Plotting the distribution of the feature and label from the Dataset.

We can see that the distributions have approximately the same shape which indicates that there is a strong linear relationship between the feature and label.
# In[97]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))

ax1.set_title('Distribution of feature X i.e. Number of Claims')
sns.distplot(data.X,bins=50,ax=ax1)

ax2.set_title('Distribution of label Y i.e. Total Payment for Corresponding claims')
sns.distplot(data.Y,bins=50,ax=ax2)

Boxplot and Violinplot gives us the quartile distribution of the feature and check for outliers.

We can see two extreme values but we will include them in our analysis.

# In[98]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))

ax1.set_ylim(-50,150)
ax1.set_title('Boxplot for X')
sns.boxplot(y='X',data=data,ax=ax1,)
sns.stripplot(y='X',color='green',data=data,jitter=True,ax=ax1,alpha=0.5)

ax2.set_ylim(-50,150)
ax2.set_title('Violinplot for X')
sns.violinplot(y='X',data=data,ax=ax2)
sns.stripplot(y='X',color='green',data=data,jitter=True,ax=ax2,alpha=0.5)


# This displays the scatter plot for Feature and Label and fits an approximate regression line for the same.

# In[99]:


fig , (ax1) = plt.subplots(1,1,figsize=(10,4))

ax1.set_title('Scatter plot between feature and Label')
sns.regplot(data=data,x='X',y='Y',ax=ax1)


# # Training Linear Regression Model
# Here we will train the Linear Regression model from scikit-learn and check the RMSE for the Training Data itself.

# In[106]:


from sklearn import  metrics
from sklearn import linear_model
X = pd.DataFrame(data.X)
Y = data.Y
regr = linear_model.LinearRegression()
regr.fit(X,Y)
Y_pred = regr.predict(X)
mse = metrics.mean_squared_error(Y_pred,Y)


# In[107]:


from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 1/3, random_state = 0)


# In[108]:


### Fitting Simple Linear Regression to the training set


# In[109]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)


# In[110]:


# Predicting the Test set result


# In[111]:


Y_Pred = regressor.predict(X_Test)


# In[112]:


# Visualising the Training set results


# In[113]:


plt.scatter(X_Train, Y_Train, color = 'red')
plt.plot(X_Train, regressor.predict(X_Train), color = 'blue')
plt.title('Total payment for all claims vs no of claims  (Training Set)')
plt.xlabel('no of claims')
plt.ylabel('total payment for all claims')
plt.show()


# In[114]:


# Visualising the Test set results


# In[115]:


plt.scatter(X_Test, Y_Test, color = 'red')
plt.plot(X_Train, regressor.predict(X_Train), color = 'blue')
plt.title('Total payment for all claims vs no of claims  (Training Set)')
plt.xlabel('no of claims')
plt.ylabel('total payment for all claims')
plt.show()


# In[127]:


regressor.score(X, Y)


# In[ ]:




