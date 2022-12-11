To create the original results of this paper with the model selected for interpretation (BERT with MLP Head), the code should be run as follows:
* model_train.py: this code downloads the data and splits the testing data from the validation and training sets
* model_test.py: this code runs the model on the withheld test data 
* model_interpret.py: this code interprets the model using the test data. The model and test data must be downloaded/created for this file to run.

To avoid re-training the model, run the code as follows:
* model_download.py: downlaods the model from Google Drive 
* model_train.py: toggle 'save_model' off; only output will be the split test data 
* model_test.py: this code runs the model on the withheld test data 
* model_interpret.py: this code interprets the model using the test data. The model and test data must be downloaded/created for this file to run.


The EDA.py file may be run regardless of order. 
