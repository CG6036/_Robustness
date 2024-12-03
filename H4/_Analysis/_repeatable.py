# Import Modules
import pandas as pd
import numpy as np
import sqlite3
from nltk import FreqDist
import pickle
import math

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

def calculate_entropy(probabilities):
    """ Calculate the Shannon entropy of a given list of probabilities. """
    entropy = 0
    for p in probabilities:
        if p > 0:
            entropy += p * math.log(p, 2)
    return -entropy


# Import Dataset
conn = sqlite3.connect('/data1/StackOverflow/stackexchange-to-sqlite/stack.db')
query = '''
SELECT id, post_type, creation_date, owner_user_id, tags, body
FROM posts
WHERE creation_date > '2021-09-01'
AND creation_date < '2023-09-01';
'''
df = pd.read_sql_query(query, conn)
conn.close()

# Load user data with user_type information
user_df = pd.read_csv("split_power_casual.csv")
user_df

# Load pre-computed LSM data
with open(file='/data1/StackOverflow/_Final/LSM_new_result.pickle', mode='rb') as f:
    lsm_og=pickle.load(f)

# Preprocess
df = df.dropna(subset=['owner_user_id'])
df['owner_user_id'] = df['owner_user_id'].astype(int).astype(str)
df['creation_date'] = pd.to_datetime(df['creation_date'])
df['year_month_day'] = df['creation_date'].dt.to_period('D')
df['year_month_day'] = df['year_month_day'].astype(str)
user_df['owner_user_id'] = user_df['owner_user_id'].astype(str)
lsm_og['year_month'] = lsm_og['year_month'].astype(str)
lsm_og = lsm_og.rename(columns={'year_month':'year_month_day'})

# Merge
df_merge = pd.merge(df, user_df[['owner_user_id', 'user_type']], on = 'owner_user_id', how = 'left')
df_merge_lsm = pd.merge(lsm_og, user_df, on='owner_user_id', how = 'left')
df_merge = df_merge.dropna(subset=['user_type'])
df_merge_lsm = df_merge_lsm.dropna(subset=['user_type'])
#df_casual = df_merge[df_merge['user_type'] == 'casual']
#df_intensive = df_merge[df_merge['user_type'] == 'intensive']
#df_top = df_merge[df_merge['user_type'] == 'top']

user_types = df_merge['user_type'].unique()
for u_type in user_types:
    # Target User Data
    data = df_merge[df_merge['user_type'] == u_type]
    questions = data[data['post_type'] == 'question']
    answers = data[data['post_type'] == 'answer']
    # Load Baseline (year_month_day, T_d, P_t)
    baseline = pd.read_csv('/data1/StackOverflow/_Final/lsm_new.csv')
    baseline = baseline[['year_month_day', 'T_d', 'P_t', 'month']]
    
    # 1) Volume Aggregation
    df_q = questions.groupby('year_month_day').size().reset_index(name = 'q')
    df_a = answers.groupby('year_month_day').size().reset_index(name = 'a')
    df_final = pd.merge(df_q, df_a, on='year_month_day', how='outer').fillna(0)
    df_final['q'] = df_final['q'].astype(int)
    df_final['a'] = df_final['a'].astype(int)
    df_final['ln_q'] = np.log(df_final['q'])
    df_final['ln_a'] = np.log(df_final['a'])
    df_final = pd.merge(baseline, df_final, on = 'year_month_day', how = 'left')
    
    # 2) Entropy
    year_month_day = data.year_month_day.unique()
    # Extract keys througout the whole data
    all_keys = tag_freq(questions)
    all_keys = pd.DataFrame(all_keys, index = ['tag']).transpose().reset_index()
    # compute tagShare on each month
    for i in range(len(year_month_day)):
        target_data = questions[questions['year_month_day'] == year_month_day[i]]
        tags = tag_freq(target_data)
        tagCount = pd.DataFrame(tags, index = ['tag']).transpose().reset_index()
        tagShare = []
        for j in range(len(tagCount)):
            tagShare.append((tagCount['tag'][j] / tagCount['tag'].sum())*100)
        tagCount['tagShare'] = tagShare
        varName = year_month_day[i].replace('-', '_')
        tagCount = tagCount.rename(columns = {'tag':f'tag_{varName}','tagShare':f'tagShare_{varName}'})
        # merge here.
        all_keys = pd.merge(all_keys, tagCount, on = 'index', how = 'left')
    # Measure score
    entropy_Score = []
    # Calculate Entropy for each monthly tag share column.
    for i in range(3, all_keys.shape[1], 2):
        arr = np.array(all_keys.iloc[:, i])
        arr = arr/100
        arrList = arr.tolist()
        entropy_Score.append(calculate_entropy(arrList))
    df_final['entropy'] = entropy_Score
    df_final['ln_entropy'] = np.log(df_final['entropy'])
    
    # 3) LSM
    data_lsm = df_merge_lsm[df_merge_lsm['user_type'] == u_type]
    # Aggregation
    lsm = data_lsm.groupby(['year_month_day'])['lsm_score'].mean().reset_index()
    df_final['lsm'] = lsm['lsm_score']
    df_final['ln_lsm'] = np.log(df_final['lsm'])

    # Save File after each iteration
    file_name = f"df_split_{u_type}.csv"
    df_final.to_csv(file_name, index=False)