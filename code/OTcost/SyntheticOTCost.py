#!/usr/bin/env python
# coding: utf-8

# In[1]:


ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/classes/otcost_fl'


# In[34]:


import numpy as np
import sys
sys.path.append(f'{ROOT_DIR}/code')
sys.path.append(f'{ROOT_DIR}/code/evaluation')
import OTCost as ot
import dataCreator as dc
import importlib
importlib.reload(ot)
importlib.reload(dc)
import warnings
warnings.filterwarnings("default", category=UserWarning)


# In[13]:


private =  False
DATASET = 'Synthetic'
SAVE = True


# ## create data OT cost 0-0.5

# In[37]:


data['1'].shape


# In[35]:


## 0.0
importlib.reload(ot)
importlib.reload(dc)
##create data
data, label = dc.non_iid_creator(frac = 0.0)
## calculate cost
Synthetic_OTCost_label = ot.OTCost(DATASET, data, label)
cost = Synthetic_OTCost_label.calculate_ot_cost()
cost = "{:.2f}".format(float(cost))

##save data
if SAVE:
    dc.saveDataset(data['1'],label['1'], f'data_1_{cost}')
    dc.saveDataset(data['2'],label['2'], f'data_2_{cost}')


# In[28]:


## 0.1
importlib.reload(ot)
importlib.reload(dc)
##create data
data, label = dc.non_iid_creator(0.21)
## calculate cost
Synthetic_OTCost_label = ot.OTCost(DATASET, data, label)
cost = Synthetic_OTCost_label.calculate_ot_cost()
cost = "{:.2f}".format(float(cost))

if SAVE:
    dc.saveDataset(data['1'],label['1'], f'data_1_{cost}')
    dc.saveDataset(data['2'],label['2'], f'data_2_{cost}')


# In[29]:


## 0.2
importlib.reload(ot)
importlib.reload(dc)
##create data
data, label = dc.non_iid_creator(0.42)
## calculate cost
Synthetic_OTCost_label = ot.OTCost(DATASET, data, label)
cost = Synthetic_OTCost_label.calculate_ot_cost()
cost = "{:.2f}".format(float(cost))

if SAVE:
    dc.saveDataset(data['1'],label['1'], f'data_1_{cost}')
    dc.saveDataset(data['2'],label['2'], f'data_2_{cost}')


# In[30]:


## 0.3
importlib.reload(ot)
importlib.reload(dc)
##create data
data, label = dc.non_iid_creator(0.61)
## calculate cost
Synthetic_OTCost_label = ot.OTCost(DATASET, data, label)
cost = Synthetic_OTCost_label.calculate_ot_cost()
cost = "{:.2f}".format(float(cost))

if SAVE:
    dc.saveDataset(data['1'],label['1'], f'data_1_{cost}')
    dc.saveDataset(data['2'],label['2'], f'data_2_{cost}')


# In[31]:


## 0.4
importlib.reload(ot)
importlib.reload(dc)
##create data
data, label = dc.non_iid_creator(0.76)
## calculate cost
Synthetic_OTCost_label = ot.OTCost(DATASET, data, label)
cost = Synthetic_OTCost_label.calculate_ot_cost()
cost = "{:.2f}".format(float(cost))
if SAVE:
    dc.saveDataset(data['1'],label['1'], f'data_1_{cost}')
    dc.saveDataset(data['2'],label['2'], f'data_2_{cost}')


# In[32]:


## 0.5
importlib.reload(ot)
importlib.reload(dc)
##create data
data, label = dc.non_iid_creator(0.87)
## calculate cost
Synthetic_OTCost_label = ot.OTCost(DATASET, data, label)
cost = Synthetic_OTCost_label.calculate_ot_cost()
cost = "{:.2f}".format(float(cost))

##save data
if SAVE:
    dc.saveDataset(data['1'],label['1'], f'data_1_{cost}')
    dc.saveDataset(data['2'],label['2'], f'data_2_{cost}')


# In[33]:


## 0.7
importlib.reload(ot)
importlib.reload(dc)
##create data
data, label = dc.non_iid_creator(1)
## calculate cost
Synthetic_OTCost_label = ot.OTCost(DATASET, data, label)
cost = Synthetic_OTCost_label.calculate_ot_cost()
cost = "{:.2f}".format(float(cost))

##save data
if SAVE:
    dc.saveDataset(data['1'],label['1'], f'data_1_{cost}')
    dc.saveDataset(data['2'],label['2'], f'data_2_{cost}')


# # Additional estimates for reviewer

# In[4]:


## IDENTICAL DATASETS
importlib.reload(ot)
importlib.reload(dc)
##create data
data, label = dc.non_iid_creator(0)
data = {'1': data['1'], '2': data['1']}
label = {'1': label['1'], '2': label['1']}
## calculate cost
Synthetic_OTCost_label = ot.OTCost(DATASET, data, label)
cost = Synthetic_OTCost_label.calculate_ot_cost()
cost = "{:.2f}".format(float(cost))


# In[15]:


## SUBSAMPLED DATASETS
importlib.reload(ot)
importlib.reload(dc)
##create data
data, label = dc.non_iid_creator(0)
subset_size = int(0.5 * len(data['1'])) 
data['2'] = data['1'][:subset_size] 
label['2'] = label['1'][:subset_size] 
## calculate cost
Synthetic_OTCost_label = ot.OTCost(DATASET, data, label)
cost = Synthetic_OTCost_label.calculate_ot_cost()
cost = "{:.2f}".format(float(cost))


# In[38]:


## OPPOSITE LABELS
importlib.reload(ot)
importlib.reload(dc)
##create data
data, label = dc.non_iid_creator(0)
label['2'] = 1 - label['2']
## calculate cost
Synthetic_OTCost_label = ot.OTCost(DATASET, data, label)
cost = Synthetic_OTCost_label.calculate_ot_cost()
cost = "{:.2f}".format(float(cost))

