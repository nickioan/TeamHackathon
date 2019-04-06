# Importing the dataset
import pandas as pd
import numpy as np
dataset = pd.read_csv('data/train.csv')
dataset = dataset.dropna()
#dataset = dataset.dropna()
all_catagories = {}
for column in dataset:
    x = dataset[column].unique()
    all_catagories[column] = x
    np.append(all_catagories[column], ['NaN'])
    mylist=all_catagories[column]
    #mylist.append('NaN')
    #all_catagories = mylist
    #all_catagories[column].append(np.array(['NaN']))
print(all_catagories)

import pickle
with open("data/all_cat_labels.pkl", "wb") as f:
    pickle.dump(all_catagories, f)


