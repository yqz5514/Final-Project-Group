# Author: Aolson
# Date: 11/18/22
# Purpose: download pre-processed data, pre-trained model from paper github to cloud environment
# Resources:

# imports/installs
import os
#os.system('pip3 install gdown')
import gdown
os.chdir('/home/ubuntu/Final-Project-Group/Code')
PATH = os.getcwd()

data_url = 'https://drive.google.com/u/0/uc?id=1BWBN1coTWGBqrWoOfRc5dhojPHhatbYs&export=download'
model_url = 'https://drive.google.com/u/0/uc?id=1H50fClyTkNprWJNh10HWdGEdDdQIkzsI&export=download'

# download and unzip pre-processed data
gdown.download(data_url, 'bert_data_xsum_final.zip', quiet=False)
os.system('unzip bert_data_xsum_final.zip')

# download and unzip pre-trained model
gdown.download(model_url, 'bertsumextabs_xsum_final_model.zip', quiet=False)
os.system('unzip bertsumextabs_xsum_final_model.zip')