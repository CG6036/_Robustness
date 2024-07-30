import pickle
import pandas as pd
import numpy as np
import sqlite3
import os
import matplotlib.pyplot as plt
################

# Import Dataset (from Entropy.ipynb)
with open(file = '/data1/StackOverflow/_Robustness/H4/DDD/df_old_cluster.pickle', mode = 'rb') as file:
    df_old = pickle.load(file)
with open(file = '/data1/StackOverflow/_Robustness/H4/DDD/df_new_cluster.pickle', mode = 'rb') as file:
    df_new = pickle.load(file)

#################

# Preprocess
from nltk import FreqDist
def wc(text):
    """
    Cleaning function to be used with our first wordcloud
    """
    
    if text:
        tags = text.replace('><',' ')
        tags = tags.replace('-','')
        tags = tags.replace('.','DOT')
        tags = tags.replace('c++','Cpp')
        tags = tags.replace('c#','Csharp')
        tags = tags.replace('>','')
        return tags.replace('<','')
    else:
        return 'None'
    
def clean_tags(text):
    """
    Cleaning function for tags
    """
    
    if text:
        tags = text.replace('><',' ')
        tags = tags.replace('>','')
        return tags.replace('<','')
    else:
        return 'None'
    
def tag_freq(data):
    tags = data['tags'].str.replace('[\["\]]', '', regex=True)
    tags = [tag for i in tags.apply(lambda x: wc(x)) for tag in i.split(', ')]
    result = FreqDist(tags)
    return result

# Preprocessing for Oldbie
df_old['creation_date'] = pd.to_datetime(df_old['creation_date'])
df_old['year_month'] = df_old['creation_date'].dt.to_period('D')
df_old['year_month'] = df_old['year_month'].astype(str)
year_month = df_old.year_month.unique()

all_keys = tag_freq(df_old)
all_keys = pd.DataFrame(all_keys, index = ['tag']).transpose().reset_index()
num_community = df_old.community.unique()
num_community.sort()

all_keys = tag_freq(df_old)
all_keys = pd.DataFrame(all_keys, index = ['tag']).transpose().reset_index()

for i in range(len(year_month)):
    data = df_old[df_old['year_month'] == year_month[i]]
    for j in range(len(num_community)):
        data_community = data[data['community']==j]
        tags = tag_freq(data_community)
        tagCount = pd.DataFrame(tags, index = ['tag']).transpose().reset_index()
        tagShare = []
        for k in range(len(tagCount)):
            tagShare.append((tagCount['tag'][k] / tagCount['tag'].sum())*100)
        tagCount['tagShare'] = tagShare
        varName = year_month[i].replace('-', '_')+f'_{j}'
        tagCount = tagCount.rename(columns = {'tag':f'tag_{varName}','tagShare':f'tagShare_{varName}'})
        all_keys = pd.merge(all_keys, tagCount, on = 'index', how = 'left')

# Save data
with open(file = 'cluster_tag_share_h4_old.pickle', mode = 'wb') as file:
    pickle.dump(all_keys, file)

# Preprocessing for Newbie
df_new['creation_date'] = pd.to_datetime(df_new['creation_date'])
df_new['year_month'] = df_new['creation_date'].dt.to_period('D')
df_new['year_month'] = df_new['year_month'].astype(str)
year_month = df_new.year_month.unique()

all_keys = tag_freq(df_new)
all_keys = pd.DataFrame(all_keys, index = ['tag']).transpose().reset_index()
num_community = df_new.community.unique()
num_community.sort()

all_keys = tag_freq(df_new)
all_keys = pd.DataFrame(all_keys, index = ['tag']).transpose().reset_index()

for i in range(len(year_month)):
    data = df_new[df_new['year_month'] == year_month[i]]
    for j in range(len(num_community)):
        data_community = data[data['community']==j]
        tags = tag_freq(data_community)
        tagCount = pd.DataFrame(tags, index = ['tag']).transpose().reset_index()
        tagShare = []
        for k in range(len(tagCount)):
            tagShare.append((tagCount['tag'][k] / tagCount['tag'].sum())*100)
        tagCount['tagShare'] = tagShare
        varName = year_month[i].replace('-', '_')+f'_{j}'
        tagCount = tagCount.rename(columns = {'tag':f'tag_{varName}','tagShare':f'tagShare_{varName}'})
        all_keys = pd.merge(all_keys, tagCount, on = 'index', how = 'left')

# Save data
with open(file = 'cluster_tag_share_h4_new.pickle', mode = 'wb') as file:
    pickle.dump(all_keys, file)
