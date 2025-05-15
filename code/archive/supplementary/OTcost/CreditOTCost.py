#!/usr/bin/env python
# coding: utf-8

# In[1]:


ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/classes/otcost_fl'
DATA_DIR = f'{ROOT_DIR}/data/Credit'


# In[7]:


import pandas as pd
import numpy as np
import copy
import sys
sys.path.append(f'{ROOT_DIR}/code')
import OTCost as ot
import importlib
importlib.reload(ot)
import random
SEED = 1234
np.random.seed(SEED)
random.seed(SEED)


# In[24]:


## create overall test set taking random sample of dataset
def fracData(data, share, share_pos = 0.1):
    ## size of dataset
    num = int(data.shape[0] * share)
    ## share of pos and neg
    pos = int(share_pos * num)
    neg = int((1-share_pos) * num)
    df = pd.concat([data.groupby('Class').get_group(0).sample(n = neg, random_state = SEED),
             data.groupby('Class').get_group(1).sample(n = pos, random_state = SEED)])
    return df.sample(frac = 1)


# In[9]:


def splitDataCredit(data, frac_pos, frac_neg):
    df_1 = pd.concat([data.groupby('Class').get_group(0).sample(frac = frac_neg, random_state = SEED),
             data.groupby('Class').get_group(1).sample(frac = frac_pos, random_state = SEED)])
    df_2 = data.loc[~data.index.isin(df_1.index)]
    return df_1.sample(frac = 1), df_2.sample(frac = 1)


# In[10]:


def splitLabel(df):
##split into features and labels
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    return X.values, y.values.reshape(-1)


# In[11]:


def dictionaryCreater(d1, d2):
    ##wrangle to dictionary for OT cost calculation
    X1, y1 = splitLabel(d1)
    X2, y2 = splitLabel(d2)
    data, label = {"1": X1, "2": X2}, {"1": y1.reshape(1,-1)[0], "2": y2.reshape(1,-1)[0]}
    return data, label


# In[12]:


def sampler(data, label, num = 400):
    data_, label_  = {}, {}
    for i in data:
        idx = np.random.choice(np.arange(data[i].shape[0]), num, replace=False)
        data_[i] = data[i][idx]
        label_[i] = label[i][idx]
    return data_, label_


# In[13]:


def addNoise(data, mean = 0, sigma = 1):
    k = data.shape[1]
    n = data.shape[0]
    noise = np.random.normal(mean, sigma, size = n*k).reshape(n,k)
    data_ = copy.deepcopy(data)
    data_.iloc[:,1:30] += noise[:, 1:30]
    return data_


# In[14]:


def saveDataset(X,y, name):
    d1= np.concatenate((X, y.reshape(-1,1)), axis=1)
    np.savetxt(f'{DATA_DIR}/{name}.csv',d1)
    return


# ## Load data

# In[26]:


##load dataset
df = pd.read_csv(f'{DATA_DIR}/creditcard.csv').drop(columns =['Time', 'Amount'])
df_train_pool = df.sample(frac=0.9, random_state=SEED)
df_test_global = df.drop(df_train_pool.index)
df_ = fracData(df, share = 0.01) # Small amount taken to mimic FL use case


# In[22]:


X_test_global, y_test_global = splitLabel(df_test_global)
saveDataset(X_test_global, y_test_global, f'data_test_global')


# ## OT cost

# In[34]:


private = False
DATASET = 'Credit'
SAVE = True


# In[35]:


importlib.reload(ot)
frac_pos, frac_neg = 0.5, 0.5
d1, d2 = splitDataCredit(df_, frac_pos, frac_neg)
data, label = dictionaryCreater(d1, d2)
data_, label_ = sampler(data, label)

Credit_OTCost_label = ot.OTCost(DATASET, data_, label_)
cost = Credit_OTCost_label.calculate_ot_cost()
cost = "{:.2f}".format(float(cost))

if SAVE:
    saveDataset(data['1'],label['1'], f'data_1_{cost}')
    saveDataset(data['2'],label['2'], f'data_2_{cost}')


# In[36]:


importlib.reload(ot)
bias = 0.25
frac_pos, frac_neg = 0.5*(1+bias), 0.5*(1-bias)
d1, d2 = splitDataCredit(df_, frac_pos, frac_neg)
data,  label = dictionaryCreater(d1, d2)
data_, label_ = sampler(data, label)

Credit_OTCost_label = ot.OTCost(DATASET, data_, label_)
cost = Credit_OTCost_label.calculate_ot_cost()
cost = "{:.2f}".format(float(cost))


if SAVE:
    saveDataset(data['1'],label['1'], f'data_1_{cost}')
    saveDataset(data['2'],label['2'], f'data_2_{cost}')


# In[37]:


importlib.reload(ot)
bias = 0.51
frac_pos, frac_neg = 0.5*(1+bias), 0.5*(1-bias)
d1, d2 = splitDataCredit(df_, frac_pos, frac_neg)
data,  label = dictionaryCreater(d1, d2)
data_, label_ = sampler(data, label)

Credit_OTCost_label = ot.OTCost(DATASET, data_, label_)
cost = Credit_OTCost_label.calculate_ot_cost()
cost = "{:.2f}".format(float(cost))


if SAVE:
    saveDataset(data['1'],label['1'], f'data_1_{cost}')
    saveDataset(data['2'],label['2'], f'data_2_{cost}')


# In[38]:


importlib.reload(ot)
bias = 0.77
frac_pos, frac_neg = 0.5*(1+bias), 0.5*(1-bias)
d1, d2 = splitDataCredit(df_, frac_pos, frac_neg)
data,  label = dictionaryCreater(d1, d2)
data_, label_ = sampler(data, label)

Credit_OTCost_label = ot.OTCost(DATASET, data_, label_)
cost = Credit_OTCost_label.calculate_ot_cost()
cost = "{:.2f}".format(float(cost))


if SAVE:
    saveDataset(data['1'],label['1'], f'data_1_{cost}')
    saveDataset(data['2'],label['2'], f'data_2_{cost}')


# In[39]:


importlib.reload(ot)
bias = 0.85
frac_pos, frac_neg = 0.5*(1+bias), 0.5*(1-bias)
d1, d2 = splitDataCredit(df_, frac_pos, frac_neg)
data,  label = dictionaryCreater(d1, d2)
data_, label_ = sampler(data, label)

Credit_OTCost_label = ot.OTCost(DATASET, data_, label_)
cost = Credit_OTCost_label.calculate_ot_cost()
cost = "{:.2f}".format(float(cost))


if SAVE:
    saveDataset(data['1'],label['1'], f'data_1_{cost}')
    saveDataset(data['2'],label['2'], f'data_2_{cost}')

