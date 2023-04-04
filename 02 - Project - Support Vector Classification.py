#!/usr/bin/env python
# coding: utf-8

# # Project - Support-Vector Classification
# - Classify dogs

# ### Classification task
# - 3 dog types and corresponding data
# 
# <table>
#     <tr>
#         <td>Dobermann</td>
#         <td><img src="img/dobermann.jpg" width="100" align="left"></td>
#     </tr>
#     <tr>
#         <td>German Shepherd</td>
#         <td><img src="img/german_shepherd.jpg" width="100" align="left"></td>
#     </tr>
#     <tr>
#         <td>Rottweiler</td>
#         <td><img src="img/rottweiler.jpg" width="100" align="left"></td>
#     </tr>
# </table>

# ### Step 1: Import libraries

# In[1]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Step 2: Read the data
# - Use Pandas [read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) method to read **files/data_02.csv**

# In[2]:


data = pd.read_csv('files/data_02.csv')


# In[3]:


data.head()


# ### Step 3: Make class IDs
# - Note: The model works with name classes - but often it is convenient to use numbers for classes
# - Create a dictionary to map class names to integers numbers.
#     - Hint: **class_ids = {'Dobermann': 0, 'German Shepherd': 1, 'Rottweiler': 2}**
# - Create a column with **Class ID**
#     - Hint: you can use **apply(lambda x: class_ids[x])**

# In[4]:


class_ids = {'Dobermann': 0, 'German Shepherd': 1, 'Rottweiler': 2}
data['Class ID'] = data['Class'].apply(lambda x: class_ids[x])


# In[5]:


data.head()


# ### Step 4: Scatter plot the data
# - Create a figure and axes (**fig, ax**) from matplotlib.pyplot (**plt**)
# - Make a scatter plot using the **Class ID** column as color.
#     - Hint: [Docs](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html) use **Weight** and **Height** for **x** and **y**, respectively.

# In[6]:


fig, ax = plt.subplots()

ax.scatter(x=data['Weight'], y=data['Height'], c=data['Class ID'])


# In[ ]:





# ### Step 5: Fit a model
# - Use the **SVC** model with **kernel='linear'**
# - Fit the model

# In[7]:


model = SVC(kernel='linear')
x=data[['Weight', 'Height']]
y=data['Class ID']
model.fit(x, y)


# In[ ]:





# ### Step 6: Map out the classification
# - Create a random selection of data
#     - HINT: use **np.random.rand(10000, 2)** and "shift" the data with ***(40, 20) + (25, 55)**
# - Predict the random selction of data
# - Create a plot with the data and predictions

# In[8]:


X_test=np.random.rand(10000, 2)
X_test=X_test*(40, 20) + (25, 55)
y_pred = model.predict(X_test)


# In[9]:


fig, ax = plt.subplots()

ax.scatter(x=X_test[:,0], y=X_test[:,1], c=y_pred, alpha=.5)


# ### Step 7 (Optional): Map with original data points
# - Make the same plot
# - Add the data points from the original dataset
#     - You might need to make a color mapping
#     - Say, **colors = ['b', 'r', 'g']** and use it on the **Class ID** column with **apply**.

# In[10]:


fig, ax = plt.subplots()

ax.scatter(x=X_test[:,0], y=X_test[:,1], c=y_pred, alpha=.5)
colors = ['b', 'r', 'g']
x=data['Weight']
y=data['Height']
c=data['Class ID'].apply(lambda x: colors[x])
ax.scatter(x=x, y=y, c=c)


# In[ ]:




