#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # cleaning of data and info (red wine)

# In[2]:


nRowsRead = 1000
df=pd.read_csv('winequality-red.csv',delimiter=';')
df


# In[49]:


df.describe()


# In[3]:


df.info()


# In[4]:


#Checking for any missing values if present 
df.isnull().sum()


# In[5]:


# Drop any rows with missing values
df = df.dropna()
#there isno null value present in the the dataset so there wont be any removing of any row from the dataset 


# In[6]:


# Check the data types of each column
print(df.dtypes)


#  

# # cleaning of data and info (white wine)

# In[7]:


df1=pd.read_csv('winequality-white.csv',delimiter=";")
df1


# In[50]:


df1.describe()


# In[8]:


df1.info()


# In[9]:


df.isnull().sum()


# In[10]:


# Drop any rows with missing values
df1 = df1.dropna()
#there isno null value present in the the dataset so there wont be any removing of any row from the dataset


# In[11]:


print(df1.dtypes)


# # the distribution of the wine quality score (RED)

# In[12]:


plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='quality', bins=7, kde=True)
plt.title('Distribution of Wine Quality(--RED--) Scores')
plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.show()


#   

# # its distribution fo wine quality (white)

# In[13]:


plt.figure(figsize=(10, 6))
sns.histplot(data=df1, x='quality', bins=7, kde=True)
plt.title('Distribution of Wine Quality( --WHITE--) Scores')
plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.show()


#  

# # representing the quality and ph value and the frequency of it in histogram

# In[41]:


plt.figure(figsize=(10,6))
sns.countplot(df['quality'])
plt.title('Distribution of Quality Scores of red wine ')
plt.show()


# In[14]:


plt.hist(df[["quality",'pH',]],color=["green",'lightblue'], ec="yellow",lw=2)

labels=('quality','pH')
colors=['green','blue']
plt.legend(labels=labels)


# # THIS IS FOR THE PAIRPLOT OF THE RED-WINE DATA

# In[15]:


sns.pairplot(df, hue='quality')
plt.legend()
plt.show()


# # the most important factors that influence the quality of wine (red)

# In[16]:


from sklearn.linear_model import LinearRegression
# Load the wine quality dataset
# Create a linear regression model
model = LinearRegression()


# In[17]:


# Fit the model to the data
model.fit(df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']], df['quality'])
# Print the model coefficients
print(model.coef_)


# # key findings 

# In[39]:


# Check the correlation between different features
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of red Wine Features')
plt.show()


# In[18]:



# Create a scatter plot of alcohol content vs. wine quality
plt.scatter(df['alcohol'], df['quality'])
plt.xlabel('Alcohol Content')
plt.ylabel('Wine Quality')
plt.show()
# Create a scatter plot of volatile acidity vs. wine quality
plt.scatter(df['volatile acidity'], df['quality'])
plt.xlabel('Volatile Acidity')
plt.ylabel('Wine Quality')
plt.show()
# Create a scatter plot of residual sugar vs. wine quality
plt.scatter(df['residual sugar'], df['quality'])
plt.xlabel('Residual Sugar')
plt.ylabel('Wine Quality')
plt.show()
# Create a scatter plot of pH vs. wine quality
plt.scatter(df['pH'], df['quality'])
plt.xlabel('pH')
plt.ylabel('Wine Quality')
plt.show()
# Create a scatter plot of density vs. wine quality
plt.scatter(df['density'], df['quality'])
plt.xlabel('Density')
plt.ylabel('Wine Quality')
plt.show()
# Create a scatter plot of color intensity vs. wine quality
plt.scatter(df['citric acid'], df['quality'])
plt.xlabel('citric acid')
plt.ylabel('Wine Quality')
plt.show()

# Create a scatter plot of color intensity vs. wine quality
plt.scatter(df['chlorides'], df['quality'])
plt.xlabel('chlorides')
plt.ylabel('Wine Quality')
plt.show()

# Create a scatter plot of sulphates vs. wine quality
plt.scatter(df['sulphates'], df['quality'])
plt.xlabel('sulphates')
plt.ylabel('Wine Quality')
plt.show()

# Create a scatter plot of sulphates vs. wine quality
plt.scatter(df['fixed acidity'], df['quality'])
plt.xlabel('fixed acidity')
plt.ylabel('Wine Quality')
plt.show()


#  

#   

#  

# ''' 
# # The key findings of the dataset are as follows for red wine :
# 
# 
# * There is a positive correlation between alcohol content and wine quality.
# * There is a negative correlation between volatile acidity and wine quality.
# * There is a negative correlation between residual sugar and wine quality.
# * There is a negative correlation between pH and wine quality.
# * There is a positive correlation between density and wine quality.
# * There is a positive correlation between Sulphates  and wine quality.
# '''

#   

#   

#    

# # THIS IS FOR THE PAIRPLOT OF THE WHITE-WINE DATA

# In[48]:


plt.figure(figsize=(10,6))
sns.countplot(df1['quality'],palette = "Set1")
plt.title('Distribution of Quality Scores of white wine ')
plt.show()


#  

# In[19]:


plt.hist(df1[["quality",'pH',]],color=["red",'lightblue'], ec="orange",lw=3)

labels=('quality','pH')
colors=['green','blue']
plt.legend(labels=labels)


#  

# In[20]:


sns.pairplot(df1, hue='quality')
plt.show()


# # the most important factors that influence the quality of wine (white)

# In[21]:


from sklearn.linear_model import LinearRegression
# Load the wine quality dataset
# Create a linear regression model
model1 = LinearRegression()


# In[22]:


# Fit the model to the data
model1.fit(df1[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']], df1['quality'])
# Print the model coefficients
print(model1.coef_)


# # The above shown coefficients represent the importance of each feature in predicting the quality of a wine.

# # below is the key feature of white wine

# In[47]:


# Check the correlation between different features
plt.figure(figsize=(10,8))
sns.heatmap(df1.corr(), annot=True, cmap='inferno')
plt.title('Correlation Matrix of white  Wine Features')
plt.show()


# In[23]:



# Create a scatter plot of alcohol content vs. wine quality
plt.scatter(df1['alcohol'], df1['quality'])
plt.xlabel('Alcohol Content')
plt.ylabel('Wine Quality')
plt.show()
# Create a scatter plot of volatile acidity vs. wine quality
plt.scatter(df1['volatile acidity'], df1['quality'])
plt.xlabel('Volatile Acidity')
plt.ylabel('Wine Quality')
plt.show()
# Create a scatter plot of residual sugar vs. wine quality
plt.scatter(df1['residual sugar'], df1['quality'])
plt.xlabel('Residual Sugar')
plt.ylabel('Wine Quality')
plt.show()
# Create a scatter plot of pH vs. wine quality
plt.scatter(df1['pH'], df1['quality'])
plt.xlabel('pH')
plt.ylabel('Wine Quality')
plt.show()
# Create a scatter plot of density vs. wine quality
plt.scatter(df1['density'], df1['quality'])
plt.xlabel('Density')
plt.ylabel('Wine Quality')
plt.show()
# Create a scatter plot of color intensity vs. wine quality
plt.scatter(df1['citric acid'], df1['quality'])
plt.xlabel('citric acid')
plt.ylabel('Wine Quality')
plt.show()

# Create a scatter plot of color intensity vs. wine quality
plt.scatter(df1['chlorides'], df1['quality'])
plt.xlabel('chlorides')
plt.ylabel('Wine Quality')
plt.show()

# Create a scatter plot of sulphates vs. wine quality
plt.scatter(df['sulphates'], df['quality'])
plt.xlabel('sulphates')
plt.ylabel('Wine Quality')
plt.show()

# Create a scatter plot of sulphates vs. wine quality
plt.scatter(df['fixed acidity'], df['quality'])
plt.xlabel('fixed acidity')
plt.ylabel('Wine Quality')
plt.show()


#  

#  

#  

# '''
# # The key findings of the dataset are as follows for white wine :
# 
# * There is a positive correlation between alcohol and wine quality.
# 
# * There is a negative correlation between volatile acidity and wine quality.
# 
# * There is a positive correlation between residual sugar and wine quality.
# 
# * There is a positive correlation between pH and wine quality.
# 
# * There is a negative correlation between density and wine quality.
# 
# * There is a positive correlation between sulphates and wine quality.
# 
# * There is a positive correlation between fixed acidity and wine quality.
# 
# '''

#  

#  

#  

#  
