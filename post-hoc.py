# Post-hoc Analysis of Test Data 

# imports
import pandas as pd
import numpy as np 
import argparse
import os

# set global vars
# parser = argparse.ArgumentParser()
# parser.add_argument("--path", default=None, type=str, required=True)  # Path of file
# args = parser.parse_args()
# PATH = args.path
PATH = '/Users/angelinaolson/Documents/Final-Project-Group-NLP/Data'
os.chdir(PATH)

# Helper Functions:
def cleanLabel(row):
    '''
    cleans function after loading from csv file
    :param row:
    :return:
    '''
    one = row.replace('[', '').replace(']', '')
    two =  list(one.split(','))
    three = [float(item) for item in two]
    return three 

# read in test predictions
test_df = pd.read_csv('test_predictions.csv')

# get classes as dict 
classes_dict = {0: 'negative', 1: 'neutral', 2: 'positive'}

# clean labels
test_df['target'] = test_df['target'].apply(cleanLabel)
test_df['pred_labels'] = test_df['pred_labels'].apply(cleanLabel)

# create column of rounded predictions 
def getRoundedPreds(row):
    new_preds = np.zeros(3)
    new_preds[np.argmax(row)] = 1
    return list(new_preds) 

test_df['rounded_preds'] = test_df['pred_labels'].apply(getRoundedPreds)



