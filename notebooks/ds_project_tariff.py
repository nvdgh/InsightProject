#!/usr/bin/env python
# coding: utf-8

# In[177]:


#import package
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score

from lightgbm import LGBMRegressor
import patsy


# In[178]:


#import the data
data = pd.read_csv("d:/git/InSightProject - Copy/data/processed/cross_import_cn.csv")
image = Image.open("d:/git\InSightProject/streamlit_folder/bike.png")
st.title("Welcome to the Prediction App of Tariff Impact on Changes in Import Value Percentage")
st.image(image, use_column_width=True)


# In[179]:


#checking the data
st.write("This is an application for knowing how much the range of import values (quantity imported time product price) change due to 2018 tariff using Causal Forest. Let's try and see!")
check_data = st.checkbox("See the simple data")
if check_data:
    st.write(data.head())
st.write("Now let's find out how much the import values when we choosing some parameters.")


# In[180]:


#input the numbers
va_y = data.va_y.mean()
va_l = data.va_l.mean()
pl_l = st.slider("What is your business's proportion of production workers?",data.pl_l.min(), data.pl_l.max(),data.pl_l.mean())
inter_y = data.inter_y.mean()
sk_l     = st.slider("What is your business's skill intensity?",data.sk_l.min(), data.sk_l.max(),data.sk_l.mean())
m_l = data.m_l.mean()
k_l = data.k_l.mean()
rental_l = data.rental_l.mean()
temp_l      = st.slider("What is your business's temporary workers intensity?",data.temp_l.min(), data.temp_l.max(),data.temp_l.mean())
it_l = data.it_l.mean()
mkt_l = data.mkt_l.mean()
outsource_l    = st.slider("What is your business's outsourcing intensity?",data.outsource_l.min(), data.outsource_l.max(),data.outsource_l.mean())
tax_l = data.tax_l.mean()
cn_mnc_ratio    = st.slider("What is your business's multinational corporation ratio in China?",data.cn_mnc_ratio.min(), data.cn_mnc_ratio.max(),data.cn_mnc_ratio.mean())


# In[181]:


# create the input array
newx = np.array([[va_y, va_l, pl_l, inter_y, sk_l, m_l, k_l, rental_l,temp_l, it_l, mkt_l, outsource_l, tax_l, cn_mnc_ratio]])


# In[182]:


# some data prep for later
formula = """
crossiv ~ Treated + va_y+ va_l+ pl_l+ inter_y+ sk_l+ m_l+ k_l+ rental_l+ temp_l+ it_l+ mkt_l+ outsource_l+ tax_l+ cn_mnc_ratio
"""


# In[183]:


# Describing statistical models and for building design matrices.
crossiv, X = patsy.dmatrices(formula, data, return_type='dataframe')
X = X.loc[:,:]
crossiv = crossiv.iloc[:,0]


# In[184]:


treatment_variable = "Treated"
treatment = X["Treated"]
Xl = X.drop(["Intercept", "Treated"], axis=1)


# In[185]:


import os
import warnings
warnings.filterwarnings('ignore')


# In[186]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor
from causalml.dataset.regression import synthetic_data
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import shap
import matplotlib.pyplot as plt

import time
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # for lightgbm to work


# In[187]:


plt.style.use('fivethirtyeight')
n_features = 25
n_samples = 10000


# In[188]:


w_multi = np.array(['treatment_A' if x==1 else 'control' for x in treatment])


# In[189]:


model_tau = LGBMRegressor(importance_type='gain')  # specify model for model_tau


# In[190]:


base_algo = LGBMRegressor()
# base_algo = XGBRegressor()
# base_algo = RandomForestRegressor()
# base_algo = LinearRegression()

slearner = BaseSRegressor(base_algo, control_name='control')
slearner.estimate_ate(Xl, w_multi, crossiv)


# In[191]:


slearner_tau = slearner.fit_predict(Xl, w_multi, crossiv)


# In[192]:


#modelling step
#import your model
errors = np.sqrt(mean_squared_error(crossiv,slearner_tau))
predictions = slearner.predict(newx)


# In[193]:


#checking prediction house price
if st.button("Run me!"):
    st.header("Your business's import values will change an amount of {}%".format(np.round(predictions*10,2)))
    st.subheader("Your range of prediction is {}% - {}%".format(np.round(predictions*10-errors,2),np.round(predictions*10+errors,2)))


# In[ ]:




