# Post-hoc Analysis of Test Data 

# imports
import pandas as pd
import numpy as np 


# Helper Functions:
def cleanLabel(row):
    '''
    cleans function after loading from csv file
    :param row:
    :return:
    '''
    remove = ['[', ']', ',', ' ']
    label = [float(i) for i in row if i not in remove]
    return label

# read in test predictions


