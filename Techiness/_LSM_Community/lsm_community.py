# LSM vr.2 (LSM using Daily Aggregation) of Community Level Analysis
## - no need to consider consistent posting users

# Multiple users Setting
import numpy as np
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def preprocess_text(text):
    # Remove punctuation and convert text to lowercase
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

def get_function_word_categories():
    # Define function words categorized by linguistic type
    return dict(
	prepositions = 
		['about',
		'across',
		'against',
		'along ',
		'around',
		'at',
		'behind',
		'beside',
		'besides',
		'by',
		'despite',
		'down',
		'during',
		'for',
		'from',
		'in',
		'inside',
		'into',
		'near',
		'of',
		'off',
		'on',
		'onto',
		'over',
		'through',
		'to',
		'toward',
		'with',
		'within',
		'without'],
	pronouns = 
		['i',
		'y',
		'you',
		'he',
		'me',
		'him',
		'my',
		'mine',
		'her',
		'hers',
		'his',
		'myself',
		'himself',
		'herself',
		'anything',
		'everything',
		'anyone',
		'everyone',
		'ones',
		'such',
		'it',
		'we',
		'they',
		'us',
		'them',
		'our',
		'ours',
		'their',
		'theirs',
		'itself',
		'ourselves',
		'themselves',
		'something',
		'nothing',
		'someone'],
	determiners = 
		['the',
		'some',
		'this',
		'that',
		'every',
		'all',
		'both',
		'one',
		'first',
		'other',
		'next',
		'many',
		'much',
		'more',
		'most',
		'several',
		'a',
		'an',
		'any',
		'each',
		'no',
		'half',
		'twice',
		'two',
		'second',
		'another',
		'last',
		'few',
		'little',
		'less',
		'least',
		'own'],
	conjunctions = 
		['and',
		'but',
		'after',
		'when',
		'as',
		'because',
		'if',
		'what',
		'where',
		'which',
		'how',
		'than',
		'or',
		'so',
		'before',
		'since',
		'while',
		'although',
		'though',
		'who',
		'whose'],
	modal_verbs = 
		['can',
		'may',
		'will',
		'shall',
		'could',
		'might',
		'would',
		'should',
		'must',
        'ought'],
	primary_verbs = 
		['am',
		'is',
        'are',
		'was',
        'were',
		'being',
        'been'
        'be',
		'do',
		'does',
        'did',
		'have',
        'has',
		'had'],
	adverbs = 
		['here',
		'there',
		'today',
		'tomorrow',
		'now',
		'then',
		'always',
		'never',
		'sometimes',
		'usually',
		'often',
		'therefore',
		'however',
		'besides',
		'moreover',
		'though',
		'otherwise',
		'else',
		'instead',
		'anyway',
		'incidentally',
		'meanwhile',
		'hardly'],
)

def calculate_fwu(text, function_words):
    # Preprocess text and calculate total word count
    text = preprocess_text(text)
    total_words = len(text.split())
    
    # Initialize vectorizer and count function words
    vectorizer = CountVectorizer(vocabulary=function_words)
    fwu_counts = vectorizer.fit_transform([text]).toarray().flatten()
    
    # Normalize function word usage by total words
    fwu_normalized = fwu_counts / total_words
    return fwu_normalized

def calculate_community_fwu(texts):
    # Calculate the community-level FWU by averaging FWU across all users' texts
    categories = get_function_word_categories()
    community_fwu = {}
    
    for category, function_words in categories.items():
        category_fwu = np.array([calculate_fwu(text, function_words) for text in texts])
        community_fwu[category] = np.mean(category_fwu, axis=0)  # Average FWU across all texts
    
    return community_fwu

def calculate_lsm_for_all_users(texts):
    # Get categorized function words
    categories = get_function_word_categories()
    
    # Calculate community FWU
    community_fwu = calculate_community_fwu(texts)
    
    # Calculate LSM scores for each user
    lsm_scores_all_users = []
    
    for i, text in enumerate(texts):
        individual_lsm_scores = {}
        for category, function_words in categories.items():
            fwu_individual = calculate_fwu(text, function_words)
            fwu_community = community_fwu[category]
            
            # Avoid division by zero by replacing zeros with a small constant
            fwu_individual = np.where(fwu_individual == 0, 1e-10, fwu_individual)
            fwu_community = np.where(fwu_community == 0, 1e-10, fwu_community)
            
            # Calculate LSM for the current category
            lsm_scores = 1 - (np.abs(fwu_individual - fwu_community) / (fwu_individual + fwu_community))
            individual_lsm_scores[category] = np.mean(lsm_scores)  # Average LSM score for the category
        
        # Calculate the overall LSM score as the mean of all category scores
        overall_lsm_score = np.mean(list(individual_lsm_scores.values()))
        individual_lsm_scores["overall"] = overall_lsm_score
        
        lsm_scores_all_users.append({"user_id": i, **individual_lsm_scores})
    
    return pd.DataFrame(lsm_scores_all_users)


# Import Modules
import sqlite3
import pickle
import os
# Import Dataset
conn = sqlite3.connect('/data1/StackOverflow/stackexchange-to-sqlite/stack.db')
query = '''
SELECT creation_date, owner_user_id, body
FROM answers
WHERE creation_date >= '2021-09-01' AND creation_date < '2023-09-01';
'''
df = pd.read_sql_query(query, conn)
conn.close()

# erase NAs
df = df.dropna(subset=['owner_user_id'])
df['owner_user_id'] = df['owner_user_id'].astype('int').astype('str')
# Add year_month_day variable
df['creation_date'] = pd.to_datetime(df['creation_date'])
df['year_month'] = df['creation_date'].dt.to_period('D')

# Import user_cluster data
df_user_cluster = pd.read_csv("df_user_cluster.csv")
# Aggregate Daily posts by users.
df = df.groupby(['owner_user_id','year_month'])['body'].agg(lambda x: '\n'.join(x)).reset_index()
# Merge with community values
df_user_cluster['owner_user_id'] = df_user_cluster['owner_user_id'].astype(str)
df_merge = pd.merge(df, df_user_cluster[['owner_user_id','community']], on = 'owner_user_id', how = 'left')
# erase na values araised while merge
df_merge = df_merge[~df_merge['community'].isna()]
df_merge['community'] = df_merge['community'].astype(int)
df_merge = df_merge.sort_values(by = 'year_month')

dates = df_merge['year_month'].unique().astype(str)


# Algo
result = pd.DataFrame(columns=['owner_user_id', 'year_month', 'body', 'community', 'lsm_score'])
for date in dates:
    df_filtered = df_merge[df_merge['year_month']==date]
    ls_communities = df_filtered['community'].unique()
    ls_communities.sort()
    for community in ls_communities:
        df_community = df_filtered[df_filtered['community'] == community]
        lsm_scores_df = calculate_lsm_for_all_users(df_community['body'])
        df_community['lsm_score'] = list(lsm_scores_df['overall'])
        result = pd.concat([result, df_community], ignore_index=True)

# Save Data
with open(file='LSM_new_community_result.pickle', mode='wb') as f:
    pickle.dump(result, f)