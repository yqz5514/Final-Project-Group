# Author: Aolson
# Date: 11/15/22
# Purpose: load data via hugging face
# Resources: Tutorial from hugging face here: https://huggingface.co/docs/datasets/load_hub
#            Dataset link: https://huggingface.co/datasets/xsum

# imports/installations
import os
#os.system('pip install datasets')
from datasets import load_dataset_builder
from datasets import get_dataset_split_names
from datasets import load_dataset
import pandas as pd

# set global vars
dataset = 'xsum' # set name

# load in dataset to get information before actually loading train, val, test splits
ds_builder = load_dataset_builder(dataset)

# Inspect dataset description
print(ds_builder.info.description)

# Inspect dataset features
print(ds_builder.info.features)

# Get all split names
print(get_dataset_split_names(dataset))

# Define a function to load the dataframe given a split, converting to pandas instead of arrow format
def loadData(dataset, split):
    df = load_dataset(dataset, split=split)
    pd_df = pd.DataFrame(df)
    return pd_df

# Load data
train = loadData(dataset, split='train')
val = loadData(dataset, split='validation')
test = loadData(dataset, split='test')



