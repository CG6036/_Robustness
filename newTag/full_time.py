# Import Modules
import pandas as pd
import numpy as np
import sqlite3
from nltk import FreqDist
import pickle
import math
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

# Import Dataset (all time)
conn = sqlite3.connect('/data1/StackOverflow/stackexchange-to-sqlite/stack.db')
query = '''
SELECT creation_date, tags
FROM questions;
'''
df_tags = pd.read_sql_query(query, conn)
conn.close()

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

# Preprocessing
df_tags['creation_date'] = pd.to_datetime(df_tags['creation_date'])
df_tags['year_month'] = df_tags['creation_date'].dt.to_period('M') # changed from D to M
df_tags['year_month'] = df_tags['year_month'].astype(str)
year_month = df_tags.year_month.unique()
post_count = df_tags['year_month'].value_counts()

df_tags['tags'] = df_tags['tags'].apply(wc)
df_tags['tags'] = df_tags['tags'].str.replace('[\["\]]', '', regex=True)

# Analysis
list_ym = df_tags[df_tags['year_month'] >= '2021-09']['year_month'].unique()

result_final = []
for ym in list_ym:
    # compare object (past tags)
    pre = df_tags[df_tags['year_month'] < ym]['tags'] 
    pre = set([tag for i in pre.apply(lambda x:wc(x)) for tag in i.split(', ')])
    # target year
    curr = df_tags[df_tags['year_month'] == ym]
    def check_new(txt):
        l_words = txt.split(', ')
        set1 = set(l_words)
        #return set1.isdisjoint(pre) # to check if they are totally disjoint
        return not set1.issubset(pre) # to check if at least one is new.
    result = curr['tags'].apply(check_new)
    result_final.extend(result)

df_final = df_tags[df_tags['year_month'] >= '2021-09']
df_final['isNew'] = result_final

agg_df = df_final.groupby('year_month')['isNew'].agg(['sum', 'count']).reset_index()
agg_df.columns = ['YearMonth', 'newPost', 'totalPost']

# Normalize
agg_df['norm_newPost'] = agg_df['newPost'] / agg_df['totalPost']

with open(file='full_time.pickle', mode='wb') as f:
    pickle.dump(agg_df, f)