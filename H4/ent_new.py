# Import Modules
import pandas as pd
import numpy as np
import sqlite3
import pandas as pd
from nltk import FreqDist
import pickle
import math

df = pd.read_csv("df_isNew")
df_tags = df[df['isNew'] == 1]
df_tags = df_tags.drop('year_month', axis=1)

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

df_tags['creation_date'] = pd.to_datetime(df_tags['creation_date'])
df_tags['year_month'] = df_tags['creation_date'].dt.to_period('D')
df_tags['year_month'] = df_tags['year_month'].astype(str)
year_month = df_tags.year_month.unique()

# Extract keys througout the whole data
all_keys = tag_freq(df_tags)
all_keys = pd.DataFrame(all_keys, index = ['tag']).transpose().reset_index()

# compute tagShare on each month
for i in range(len(year_month)):
    data = df_tags[df_tags['year_month'] == year_month[i]]
    tags = tag_freq(data)
    tagCount = pd.DataFrame(tags, index = ['tag']).transpose().reset_index()
    tagShare = []
    for j in range(len(tagCount)):
        tagShare.append((tagCount['tag'][j] / tagCount['tag'].sum())*100)
    tagCount['tagShare'] = tagShare
    varName = year_month[i].replace('-', '_')
    tagCount = tagCount.rename(columns = {'tag':f'tag_{varName}','tagShare':f'tagShare_{varName}'})
    # merge here.
    all_keys = pd.merge(all_keys, tagCount, on = 'index', how = 'left')

# Save Data
with open(file = 'tagShare_new.pickle', mode = 'wb') as file:
    pickle.dump(all_keys, file)