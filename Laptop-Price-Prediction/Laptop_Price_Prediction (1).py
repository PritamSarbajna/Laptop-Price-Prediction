#!/usr/bin/env python
# coding: utf-8

# In[497]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# In[498]:


laptop_data = pd.read_csv('laptop_price.csv', encoding="latin-1")


# In[499]:


laptop_data = laptop_data.drop(["laptop_ID", "Product"], axis=1)


# In[500]:


laptop_data.info()


# In[501]:


laptop_data


# In[502]:


laptop_data = laptop_data.join(pd.get_dummies(laptop_data.Company))
laptop_data = laptop_data.drop("Company", axis=1)


# In[503]:


laptop_data = laptop_data.join(pd.get_dummies(laptop_data.TypeName))
laptop_data = laptop_data.drop("TypeName", axis=1)


# In[504]:


laptop_data["ScreenResolution"] = laptop_data.ScreenResolution.str.split(" ").apply(lambda x:x[-1])


# In[505]:


laptop_data["laptop_height"] = laptop_data["ScreenResolution"].str.split("x").apply(lambda x:x[0]).astype("int")
laptop_data["laptop_width"] = laptop_data["ScreenResolution"].str.split("x").apply(lambda x:x[1]).astype("int")


# In[506]:


laptop_data = laptop_data.drop(columns="ScreenResolution")


# In[507]:


laptop_data["Weight"] = laptop_data.Weight.str.split("kg").apply(lambda x:x[0]).astype("float")


# In[508]:


laptop_data["Storage_Type"] = laptop_data.Memory.str.split(" ").apply(lambda x:x[-1])


# In[509]:


laptop_data["Ram"] = laptop_data.Ram.str.split("GB").apply(lambda x:x[0]).astype("int")


# In[510]:


laptop_data["Memory"] = laptop_data.Memory.str.split(" ").apply(lambda x:x[0])


# In[511]:


laptop_data["Storage_Type"] = laptop_data.Storage_Type.apply(lambda x:x[-2:])


# In[512]:


import re
laptop_data['Memory'] = laptop_data['Memory'].apply(lambda x: re.findall(r'\d+', x)).apply(lambda x:x[0])


# In[513]:


laptop_data


# In[514]:


laptop_data = laptop_data.join(pd.get_dummies(laptop_data.Storage_Type))
laptop_data = laptop_data.drop("Storage_Type", axis=1)


# In[515]:


laptop_data["Cpu_frequency"] = laptop_data.Cpu.str.split(" ").apply(lambda x:x[-1])
laptop_data["Cpu_brand"] = laptop_data.Cpu.str.split(" ").apply(lambda x:x[0])
laptop_data = laptop_data.drop("Cpu", axis=1)


# In[516]:


laptop_data["Cpu_frequency"] = laptop_data.Cpu_frequency.str.split("GHz").apply(lambda x:x[0]).astype("float")


# In[517]:


laptop_data.drop(1191, inplace=True)


# In[518]:


laptop_data = laptop_data.reset_index(drop=True)


# In[519]:


laptop_data = laptop_data.join(pd.get_dummies(laptop_data.Cpu_brand))
laptop_data = laptop_data.drop("Cpu_brand", axis=1)


# In[520]:


laptop_data["Gpu_brand"] = laptop_data.Gpu.str.split(" ").apply(lambda x:x[0])
laptop_data = laptop_data.drop("Gpu", axis=1)


# In[521]:


laptop_data = laptop_data.join(pd.get_dummies(laptop_data.Gpu_brand, prefix='gpu'))
laptop_data = laptop_data.drop("Gpu_brand", axis=1)


# In[522]:


laptop_data["Memory"] = laptop_data["Memory"].astype("int")


# In[523]:


laptop_data = laptop_data.join(pd.get_dummies(laptop_data.OpSys, prefix='OpSys'))
laptop_data = laptop_data.drop("OpSys", axis=1)


# In[524]:


laptop_data.to_csv("cleaned_laptop_data.csv", index=False)


# In[525]:


pc_data = pd.read_csv("cleaned_laptop_data.csv")


# In[526]:


X = pc_data.drop('Price_euros', axis=1)
y = pc_data["Price_euros"]


# In[539]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=72, test_size=0.2)


# In[540]:


rf_regressor = RandomForestRegressor()


# In[541]:


rf_regressor.fit(X_train, y_train)


# In[542]:


y_pred = rf_regressor.predict(X_test)


# In[545]:


# Plotting the predicted values vs actual values
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Plotting the 45-degree line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest Regression: Actual vs Predicted')
plt.show()


# In[544]:


r2_score(y_test, y_pred)


# In[ ]:




