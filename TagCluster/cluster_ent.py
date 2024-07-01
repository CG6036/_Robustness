### 1) Merge processed community data with original data to get original tag data.

# Import Modules
import pandas as pd
import numpy as np
import sqlite3
import pandas as pd
from nltk import FreqDist
import pickle
import math
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

# Import Dataset
conn = sqlite3.connect('/data1/StackOverflow/stackexchange-to-sqlite/stack.db')
query = '''
SELECT id, creation_date, tags
FROM questions
WHERE creation_date > '2021-09-01'
AND creation_date < '2023-09-01';
'''
df_tags = pd.read_sql_query(query, conn)
conn.close()

with open(file = 'post_cluster_pre.pickle', mode = 'rb') as file:
    data = pickle.load(file)

df_tags = pd.merge(data, df_tags, on = 'id')[['id', 'creation_date_x', 'body', 'tags_y', 'community']]
df_tags.columns = ['id', 'creation_date', 'body', 'tags', 'community']

# 1) Preprocessing

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
# Preprocessing
df_tags['creation_date'] = pd.to_datetime(df_tags['creation_date'])
df_tags['year_month'] = df_tags['creation_date'].dt.to_period('D')
df_tags['year_month'] = df_tags['year_month'].astype(str)
year_month = df_tags.year_month.unique()
num_community = df_tags.community.unique()
num_community.sort()

# Extract keys througout the whole data
all_keys = tag_freq(df_tags)
all_keys = pd.DataFrame(all_keys, index = ['tag']).transpose().reset_index()

for i in range(len(year_month)):
    data = df_tags[df_tags['year_month'] == year_month[i]]
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

# Save Data
with open(file = 'cluster_tag_share.pickle', mode = 'wb') as file:
    pickle.dump(all_keys, file)