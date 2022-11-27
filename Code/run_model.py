# Author: Aolson
# Date: 11/27/22
# Purpose: explore preprocessed data
# Resources: https://github.com/nlpyang/PreSumm
import os
#os.system('git clone https://github.com/nlpyang/PreSumm.git')
BERT_DATA_PATH = os.getcwd() + '/bert_data_xsum_new'
MODEL_PATH = os.getcwd() + '/model_step_30000.pt'

# all files for training should be in folder PreSumm
os.system(f'python3 PreSumm/src/train.py -task abs -mode validate -batch_size 3000 -test_batch_size 500 -bert_data_path {BERT_DATA_PATH} -log_file ../logs/val_abs_bert_cnndm -model_path {MODEL_PATH} -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -min_length 20 -max_length 100 -alpha 0.9 -result_path ../logs/abs_bert_cnndm')
