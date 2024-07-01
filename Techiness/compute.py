import pandas as pd
from bs4 import BeautifulSoup
import pickle
from multiprocessing import Pool, cpu_count

# Load Data
with open(file = '/data1/StackOverflow/_Robustness/TagCluster/post_cluster_pre.pickle', mode = 'rb') as file:
    ques_df = pickle.load(file)
with open(file = '/data1/StackOverflow/_Robustness/TagCluster/ans_cluster_pre.pickle', mode = 'rb') as file:
    ans_df = pickle.load(file)

# Function to clean HTML tags from a string
def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text()

# Function to extract code block text
def extract_code_blocks(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    code_blocks = soup.find_all(['code', 'pre'])
    code_texts = list(set([block.get_text() for block in code_blocks]))
    return "\n".join(code_texts)

# Function to apply multiprocessing
def apply_multiprocessing(data, func, num_processors=None):
    if num_processors is None:
        num_processors = cpu_count()
    
    with Pool(num_processors) as pool:
        result = pool.map(func, data)
    
    return result

# Apply the function to the 'body' column in both dataframes
ques_df['full_body'] = apply_multiprocessing(ques_df['body'], clean_html, 8)
ans_df['full_body'] = apply_multiprocessing(ans_df['body'], clean_html, 8)
ques_df['code_body'] = apply_multiprocessing(ques_df['body'], extract_code_blocks, 8)
ans_df['code_body'] = apply_multiprocessing(ans_df['body'], extract_code_blocks, 8)

# Save Data
with open(file = 'ques_df_pre.pickle', mode = 'wb') as file:
    pickle.dump(ques_df, file)
with open(file = 'ans_df_pre.pickle', mode = 'wb') as file:
    pickle.dump(ans_df, file)