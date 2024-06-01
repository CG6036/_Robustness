### Elbow Method Experiment ###

# Import Modules
import pandas as pd
import numpy as np
import pickle
import math
import sqlite3
from nltk import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Import Data
with open('/data1/StackOverflow/Tag_Analysis/df_tags_2023.pickle', 'rb') as fr:
    df_tags = pickle.load(fr)

# Slice preGPT tags.
preGPT = df_tags[(df_tags['creation_date'] > '2021-09-01') & 
        (df_tags['creation_date'] < '2022-12-01')]

# Preprocess
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
# Tags Preprocessing
preGPT['tags'] = preGPT['tags'].str.replace('[\["\]]', '', regex=True)
preGPT['tags'] = preGPT['tags'].str.replace('c#','Csharp')
preGPT['tags'] = preGPT['tags'].str.replace('c++','Cpp')
preGPT['tags'] = preGPT['tags'].str.replace('.','DOT')
preGPT['tags'] = preGPT['tags'].str.replace('><',' ')
preGPT['tags'] = preGPT['tags'].str.replace('>','')
preGPT['tags'] = preGPT['tags'].str.replace('-','')
preGPT = preGPT.reset_index(drop = True)

# Vectorize tags
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '))
co_occurrence_matrix = vectorizer.fit_transform(preGPT['tags']).T * vectorizer.fit_transform(preGPT['tags'])

# Measure Similarity
similarity_matrix = cosine_similarity(co_occurrence_matrix)

# Assuming you want to create, for example, 2 clusters
K_range = range(1,21)
inertia = []
for K in K_range:
    kmeans = KMeans(n_clusters = K, random_state = 42)
    kmeans.fit(similarity_matrix)
    inertia.append(kmeans.inertia_)

with open('elbow.pkl','wb') as file:
    pickle.dump(inertia, file)