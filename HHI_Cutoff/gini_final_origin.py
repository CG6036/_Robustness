# Import Modules
import pandas as pd
import numpy as np
import sqlite3
from nltk import FreqDist
import pickle
import math
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

with open('/data1/StackOverflow/Tag_Analysis/df_tags_2023.pickle', 'rb') as fr:
    df_tags = pickle.load(fr)
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

# load pickle
import pickle
with open('/data1/StackOverflow/diff_in_diff/daily_tagShare_modified.pickle', 'rb') as fr:
    all_keys = pickle.load(fr)

# Split into counts and share vals.
df_count = all_keys.iloc[:,0::2]
df_share = all_keys.drop('tag', axis = 1)
df_share = df_share.iloc[:,0::2]

# df_count로 share 만들어야 함.
tagTrend = {'tagName' : df_count['index'], 'preGPT':
              df_count.iloc[:, 641:731].sum(axis = 1, skipna = True),
              'postGPT' : df_count.iloc[:, 731:].sum(axis = 1, skipna = True)}
tagTrend = pd.DataFrame(tagTrend)
tagTrend['pre_share'] = tagTrend['preGPT'] / tagTrend['preGPT'].sum()
tagTrend['post_share'] = tagTrend['postGPT'] / tagTrend['postGPT'].sum()

# Convert nan to 0
#tagTrend.fillna(0, inplace = True)
tagTrend['diff'] = tagTrend['post_share'] - tagTrend['pre_share']
tagTrend.sort_values('diff', ascending = False)

# Calcualte Gini-Coeff.
def calculate_gini(shares):
    shares = sorted(shares)
    size = len(shares)
    total_sum = sum(shares)
    abs_diffs = 0
    for i in range(size):
        for j in range(size):
            abs_diffs += abs(shares[i]-shares[j])
    gini_coeff = abs_diffs / (2 * size * total_sum)
    return gini_coeff

cutoff_List = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
result_vr6 = pd.DataFrame(columns = ['cut_off_percentage', 'coef', 'std_err', 'P_val', 'volume_outlier'])

for i in range(len(cutoff_List)):
    # filter data without outliers
    cleaned_data = tagTrend.sort_values('pre_share', ascending = False)[cutoff_List[i]:]
    cleansed = all_keys[all_keys['index'].isin(cleaned_data['tagName'])].reset_index(drop = True)
    
    # Entropy score
    Gini_coeff = []
    for j in range(3, cleansed.shape[1], 2):
        arr = cleansed.iloc[:,j]
        arr = arr[~np.isnan(arr)]
        Gini_coeff.append(calculate_gini(arr))
    result_entropy = pd.DataFrame({'year_month':year_month, 'gini_Score':Gini_coeff})
    
    # Model Construction
    entropy = list(result_entropy[(result_entropy['year_month'] > '2021-08-31') &
            (result_entropy['year_month'] < '2023-09-01')].reset_index().gini_Score) # fixed datetime
    # Split Data
    control_data = pd.DataFrame({'entropy' : entropy[:365],
                'T_d': [0]*len(entropy[:365]),
                'P_t' : [0]*90 + [1]*275})
    treated_data = pd.DataFrame({'entropy' : entropy[365:],
                'T_d': [1]*len(entropy[365:]),
                'P_t' : [0]*90 + [1]*275})
    df_did = pd.concat([control_data, treated_data], axis = 0).reset_index(drop = True)
    # Add date and month feature
    df_did['date'] = result_entropy[(result_entropy['year_month'] > '2021-08-31') &
            (result_entropy['year_month'] < '2023-09-01')].reset_index().year_month
    df_did['month'] = pd.to_datetime(df_did['date']).dt.month
    # Apply log
    df_did['ln_y'] = np.log(df_did['entropy'])

    # Result Appending
    filename = f"/data1/StackOverflow/_Robustness/HHI_Cutoff/gini_pickle_origin/entropy_{cutoff_List[i]}.pkl"
    model2 = sm.ols('ln_y ~ T_d + P_t + T_d * P_t + C(month)', df_did).fit(cov_type='HC3')
    with open(filename, "wb") as file:
        pickle.dump(model2, file)
    result_model_bottomK = pd.DataFrame({'cut_off_percentage':[cutoff_List[i]], 'coef':[model2.params['T_d:P_t']], 'std_err':[model2.bse['T_d:P_t']], 'P_val':[model2.pvalues['T_d:P_t']],
                                         'volume_outlier': [tagTrend.sort_values('pre_share', ascending = False)[:cutoff_List[i]].pre_share.sum()]})
    result_vr6 = pd.concat([result_vr6, result_model_bottomK], ignore_index = True)

with open('/data1/StackOverflow/_Robustness/HHI_Cutoff/gini_pickle_origin/result.pkl', 'wb') as file:
    pickle.dump(result_vr6, file)