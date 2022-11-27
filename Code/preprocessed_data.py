# Author: Aolson
# Date: 11/18/22
# Purpose: explore preprocessed data
# Resources:
import os
import torch

os.chdir('/home/ubuntu/Final-Project-Group/Code')
DATA_PATH = os.getcwd() + '/bert_data_xsum_new/xsum.train.0.bert.pt'

# use one example
train_batch1 = torch.load(PATH)
