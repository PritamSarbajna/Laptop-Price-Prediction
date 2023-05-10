#!/usr/bin/env python
# coding: utf-8

# In[411]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[412]:


laptop_data = pd.read_csv('laptop_price.csv', encoding="latin-1")


# In[413]:


laptop_data = laptop_data.drop(["laptop_ID", "Product"], axis=1)


# In[414]:


laptop_data.info()


# In[415]:


laptop_data


# In[416]:


laptop_data = laptop_data.join(pd.get_dummies(laptop_data.Company))
laptop_data = laptop_data.drop("Company", axis=1)


# In[417]:


laptop_data = laptop_data.join(pd.get_dummies(laptop_data.TypeName))
laptop_data = laptop_data.drop("TypeName", axis=1)


# In[418]:


laptop_data["ScreenResolution"] = laptop_data.ScreenResolution.str.split(" ").apply(lambda x:x[-1])


# In[419]:


laptop_data["laptop_height"] = laptop_data["ScreenResolution"].str.split("x").apply(lambda x:x[0]).astype("int")
laptop_data["laptop_width"] = laptop_data["ScreenResolution"].str.split("x").apply(lambda x:x[1]).astype("int")


# In[420]:


laptop_data = laptop_data.drop(columns="ScreenResolution")


# In[421]:


laptop_data["Weight"] = laptop_data.Weight.str.split("kg").apply(lambda x:x[0]).astype("float")


# In[422]:


laptop_data["Storage_Type"] = laptop_data.Memory.str.split(" ").apply(lambda x:x[-1])


# In[423]:


laptop_data["Ram"] = laptop_data.Ram.str.split("GB").apply(lambda x:x[0]).astype("int")


# In[424]:


laptop_data["Memory"] = laptop_data.Memory.str.split(" ").apply(lambda x:x[0])


# In[425]:


laptop_data["Storage_Type"] = laptop_data.Storage_Type.apply(lambda x:x[-2:])


# In[426]:


import re
laptop_data['Memory'] = laptop_data['Memory'].apply(lambda x: re.findall(r'\d+', x)).apply(lambda x:x[0])


# In[427]:


laptop_data


# In[428]:


laptop_data = laptop_data.join(pd.get_dummies(laptop_data.Storage_Type))
laptop_data = laptop_data.drop("Storage_Type", axis=1)


# In[429]:


laptop_data["Cpu_frequency"] = laptop_data.Cpu.str.split(" ").apply(lambda x:x[-1])
laptop_data["Cpu_brand"] = laptop_data.Cpu.str.split(" ").apply(lambda x:x[0])
laptop_data = laptop_data.drop("Cpu", axis=1)


# In[430]:


laptop_data["Cpu_frequency"] = laptop_data.Cpu_frequency.str.split("GHz").apply(lambda x:x[0]).astype("float")


# In[431]:


laptop_data.drop(1191, inplace=True)


# In[432]:


laptop_data = laptop_data.reset_index(drop=True)


# In[433]:


laptop_data = laptop_data.join(pd.get_dummies(laptop_data.Cpu_brand))
laptop_data = laptop_data.drop("Cpu_brand", axis=1)


# In[434]:


laptop_data["Gpu_brand"] = laptop_data.Gpu.str.split(" ").apply(lambda x:x[0])
laptop_data = laptop_data.drop("Gpu", axis=1)


# In[435]:


laptop_data = laptop_data.join(pd.get_dummies(laptop_data.Gpu_brand, prefix='gpu'))
laptop_data = laptop_data.drop("Gpu_brand", axis=1)


# In[436]:


laptop_data["Memory"] = laptop_data["Memory"].astype("int")


# In[437]:


laptop_data = laptop_data.join(pd.get_dummies(laptop_data.OpSys, prefix='OpSys'))
laptop_data = laptop_data.drop("OpSys", axis=1)


# In[438]:


laptop_data.to_csv("cleaned_laptop_data.csv", index=False)


# In[439]:


pc_data = pd.read_csv("cleaned_laptop_data.csv")


# In[440]:


X = pc_data.drop('Price_euros', axis=1)
y = pc_data["Price_euros"]


# In[441]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=956, test_size=0.2)


# In[442]:


reg = LinearRegression()


# In[443]:


reg.fit(X_train, y_train)


# In[444]:


y_pred = reg.predict(X_test)


# In[445]:


r2_score(y_test, y_pred)


# In[ ]:




