#!/usr/bin/env python
# coding: utf-8

# 1 House Price Prediction-Task-1

# In[1]:


'''Use a Dataset that includes information abouthousing prices and features like squarefootage, number of bedrooms, etc. to train amodel that can predict the price of a new house'''


# 'Data Collection: Gather a dataset that includes information about housing prices along with relevant features such as the number of bedrooms,bathrooms, square footage, location Amenities, etc.You can obtain this data from real estate websites,governmentdatabases,or other sources.'
# 
# 'Model selection : choose an appropriate machine learning algorithm for regression tasks.common choices include linear regression,decision trees,randomforests,gradient boosting,or neural networks.Experiment with different algorithms to find the one that performs best for your dataset.'
# 
# 'Monitoring and Maintenance : Continuously monitor the models performance over time and retrain it periodically with updates data to ensure its accuracy remains heigh.'

# In[2]:


'''
panda,numpy,matplotlib,seaborn,sklearn are the basic libraries used in the

email spam filtering

natural language tool kit used to study the data which means a mail

and visualized the data in the different graphical form(pictorial representation

and here we are using the linear regression to predict the price of a new house

'''


# 2. Lets See

# In[3]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use(['ggplot'])

from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.linear_model import LinearRegression


# In[4]:


#reading the csv file using the pandas

file_path="D:\\kc_house_data.csv"
df=pd.read_csv(file_path)
df


# In[5]:


df.head(10)


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.info(verbose=False)


# In[9]:


df.dtypes


# In[10]:


df.describe(include='all').T


# 3.Data - Wrangling

# In[11]:


df.drop(['id'],axis=1,inplace=True)


# In[12]:


df.head()


# In[13]:


le=LabelEncoder()
df["date"]=le.fit_transform(df["date"])
df['date'].dtype


# 4.Exploratory Data Analysis

# In[14]:


# count the number of houses with unique floor values.
df['floors'].value_counts().to_frame()


# In[15]:


df.hist(bins=50,figsize=(15,15))
plt.show()


# In[16]:


#Determine whether houses with a waterfront view or without a waterfront viewhave more price outliers.
sns.boxplot(data=df,x=df['waterfront'],y=df['price'])


# In[17]:


#determine if the feature sqft_above is negatively or positively correlated with price.
sns.regplot(data=df,x=df['sqft_above'],y=df['price'])


# In[18]:


sns.boxplot(data=df,x=df['sqft_basement'],y=df['price'])


# In[19]:


sns.barplot(data=df,x=df['floors'],y=df['price'])


# In[20]:


sns.histplot(data=df, x=df['grade'],y=df['price'] )


# In[21]:


sns.barplot(data=df,x=df['grade'],y=df['price'])


# In[22]:


corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix,
annot=True,
linewidths=0.5,
fmt=".2f",
cmap="YlGnBu");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# In[ ]:


df.skew()


# In[ ]:


df.drop('price',axis=1).corrwith(df.price).
   plot(kind='bar',grid=True,figsize=(10,6),title="Correlation with price ")


# 5.Spliting the Data Set

# In[ ]:


#to use the linear regression we need to split the given data in x and y format
x=np.array(df.drop(columns='price'))
y=np.array(df.drop(columns='price'))
space=df['sqft_living']
price=df['price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.
 ↪25,random_state=44)

print(f"the shape of x_train is : {x_train.shape}")
print(f'the shape of x_test is : {x_test.shape}')
print(f'the shape of y_tain is : {y_train.shape}')
print(f'the shape of y_test is {y_test.shape}')


# In[ ]:


#using the linear regression model

model3=LinearRegression()
model3.fit(x_train,y_train)
y_pred3=model3.predict(x_test)

print(f'R2 Score is : {r2_score(y_test,y_pred3)}')
print(f'Mae is : {mean_absolute_error(y_test,y_pred3)}')


# In[24]:


x_train=[0.00,0.25,0.50,0.1]
y_train=[0.00,0.25,0.50,0.1]

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, y_train, color='blue')
plt.title("visualization--")
plt.xlabel('space')
plt.ylabel('price')

plt.show()


# In[25]:


plt.scatter(x_train, y_train, label='Actual data',color='blue')
plt.plot(x_train, y_train, color='red')
plt.title("visualization")
plt.xlabel('space')
plt.ylabel('price')

plt.show()


# In[ ]:




