import os, itertools
import sys
# Append the directory containing the LSM module to the Python path
sys.path.append('/data1/StackOverflow/language-style-matching-python')
from lib import LSM
import pandas as pd
import numpy as np
import pickle

with open('/data1/StackOverflow/_Robustness/UserCluster/year_month.pickle', 'rb') as fr:
    year_month = pickle.load(fr)

# (Bottleneck)
# Load files in the folder.
for i in range(len(year_month)):
	for root, dirs, files in os.walk(f'/data1/StackOverflow/_Robustness/H4/DDD/lsm/lsm_data/oldbie/{year_month[i]}'): # parameter (change 'Final')
		folks = {}
		for folk in files:
			with open(os.path.join(root, folk), 'r') as f:
				folks[folk] = LSM(f.read()) # load via LSM class.

	combos = itertools.combinations(folks.items(), 2) 
	compares = []
	everybody = sum(folks.values())
	for obj1, obj2 in [combo for combo in combos]:
		compares.append([obj1[0], #obj2[0], 
			#str(obj1[1].compare(obj2[1])), # compare one-to-one.
			str(obj1[1].compare(everybody))]) # compare one-to-avg.


	col = ['User1', 'Similarity_toAvg']
	df = pd.DataFrame(compares, columns = col)
	print(f"{i} out of {len(year_month)} has been processed")

	# save via pickle
	with open(f'/data1/StackOverflow/_Robustness/H4/DDD/lsm/lsm_result/oldbie/{year_month[i]}.pickle', 'wb') as fw: # parameter (change 'Final_pickle')
		pickle.dump(df, fw)