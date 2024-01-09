#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import required libraries.
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
iris = pd.read_csv(r"C:\Users\nazee\OneDrive\Desktop\Nexus\Iris.csv") # loading the Data
df=iris.drop(['Id'],axis=1) # dropping the Id column as it won't help us in any prediction.

print(df)


# In[2]:


df.shape #to get the shape of the dataset


# In[3]:


df.info() #to get columns and their data types


# In[4]:


df.describe() #to get quick statistical summary of the dataset


# In[5]:


columns=df.columns #to get column names
print(columns)


# In[6]:


df['Species'].value_counts() #the species contains equal no.of rows


# In[7]:


sns.pairplot(df,hue='Species') #Pair plot is used to visualize the relationship between each type of column variable
plt.show()
#according to the plot we can analyse that - the iris-setosa species is clearly separated from the other two flowers.
#In petal length and petal width plots, comparitively the overlapping of classes is low.


# In[8]:


for i in columns:
  if(i!="Species"):
     abcd=sns.FacetGrid(df,hue="Species").map(sns.distplot,i).add_legend()
  plt.show()

#From the plots, we can observe that –

#In the case of Sepal Length, there is a huge amount of overlapping.
#In the case of Sepal Width also there is a huge amount of overlapping.
#In the case of Petal Length, there is a very little amount of overlapping.
#In the case of Petal Width also, there is a very little amount of overlapping.

#So we can use Petal Length and Petal Width as the classification feature.


# In[9]:


df.corr(method='pearson')#to show correlation coefficients between variables.

#from the analysis Petal length and petal width have high positive correlation of 0.96.


# In[10]:


def graph(y):
	sns.boxplot(x="Species", y=y, data=df)

plt.figure(figsize=(10,10))


plt.subplot(221)
graph('SepalLengthCm')

plt.subplot(222)
graph('SepalWidthCm')

plt.subplot(223)
graph('PetalLengthCm')

plt.subplot(224)
graph('PetalWidthCm')

plt.show()

#From the graph, we can observe that –

#Species Iris-Setosa has the smallest features and less distributed.
#Species Iris-Versicolor has the average features.
#Species Iris-Virginica has the highest features.


# In[11]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() # Label Encoding refers to converting the Non numeric form labels (i.e Species) into numeric form (i.e machine-readable form).
df['Species'] = le.fit_transform(df['Species']).copy() # Here 0 is Iris-Setosa
df.head()                                              # Here 1 is Iris-Versicolor
                                                       # Here 2 is Iris-Virginica


# In[12]:


from sklearn.model_selection import train_test_split
# train - 70
# test - 30
X = df.drop(columns=['Species']) #input attributes
Y = df['Species'] #output attribute
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30) #splits the data for training and testing (here we are splitting 70% data for training and 30% for testing)


# In[13]:


# logistic regression
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(x_train, y_train)  #training the model with the data
y_pred = model1.predict(x_test)
print(classification_report(y_test, y_pred))


# In[14]:


# knn - k-nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier()
model2.fit(x_train, y_train) #training the model with the data
y_pred = model2.predict(x_test)
print(classification_report(y_test, y_pred))


# In[15]:


# decision tree
from sklearn.tree import DecisionTreeClassifier
model3 = DecisionTreeClassifier()
model3.fit(x_train, y_train) #training the model with the data
y_pred = model3.predict(x_test)
print(classification_report(y_test, y_pred))


# In[16]:


# In this project-
# we have learnt on how to train machine learning classification model for iris flower dataset. 
#We also learned about data analysis, visualizations, data transformation, model creation, etc.,



# In[ ]:




