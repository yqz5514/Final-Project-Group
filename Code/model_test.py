import os
import torch
import pandas as pd
import regex as re
import torch.nn as nn
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from sklearn.metrics import accuracy_score
import numpy as np
import gdown


# %% -------------------------------------- Global Vars ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
OUTPUT_DIM = 3
BATCH_SIZE = 64
#MAX_VOCAB_SIZE = 25_000
MAX_LEN = 300
no_layers = 3
hidden_dim = 256
clip = 5
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model_type = 'MLP'
export_data = True

# %% -------------------------------------- Helper Functions ------------------------------------------------------------------
def TextCleaning(text):
    '''
    Takes a string of text and performs some basic cleaning.
    1. removes tabs
    2. removes newlines
    3. removes special chars
    4. creates the word "not" from words ending in n't
    '''
    # Step 1
    pattern1 = re.compile(r'\<.*?\>')
    s = re.sub(pattern1, '', text)

    # Step 2
    pattern2 = re.compile(r'\n')
    s = re.sub(pattern2, ' ', s)

    # Step 3
    pattern3 = re.compile(r'[^0-9a-zA-Z!/?]+')
    s = re.sub(pattern3, ' ', s)

    # Step 4
    pattern4 = re.compile(r"n't")
    s = re.sub(pattern4, " not", s)

    return s

def getLabel(df, label_col, input_col):
    encoded = pd.get_dummies(df, columns=[label_col])
    encoded_val = encoded.iloc[:, 1:].apply(list, axis=1)
    encoded['target'] = encoded_val
    return_df = encoded[[input_col, 'target']]
    return return_df

def create_data_loader(df, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    ds = nlpDataset(
        data_frame = df,
        tokenizer = tokenizer,
        max_len = max_len,
        input_col = input_col)
    return DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=data_collator, drop_last=True)


# %% -------------------------------------- Dataset Class ------------------------------------------------------------------
class nlpDataset(Dataset):
    def __init__(self, data_frame, tokenizer, max_len, input_col):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input = self.data_frame.iloc[idx][input_col]
        target = self.data_frame.iloc[idx]['target']
        encoding = self.tokenizer(
            input,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
            padding=True,
            return_tensors='pt', )

        output = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(target, dtype=torch.long)
        }

        return output

def Tester(model_type=model_type):
    if model_type == 'RNN':
        model = BERT_PLUS_RNN(bert, no_layers, hidden_dim, OUTPUT_DIM, BATCH_SIZE)
        model.load_state_dict(torch.load('model_nn.pt', map_location=device))
        model.to(device)

        criterion = torch.nn.BCELoss()

        # Tracking variables

        test_losses = []
        test_acc = []
        final_pred_labels = []
        final_real_labels = []
        model.eval()

        for batch in test_loader:
            with torch.no_grad():
                inputs, labels, attention_mask = batch['input_ids'].to(device), batch['labels'].to(device), batch[
                    'attention_mask'].to(device)
                test_h = model.init_hidden(len(inputs))
                test_h = tuple([each.data for each in test_h])
                output, test_h = model(inputs, test_h, attention_mask)
                test_loss = criterion(output, labels.float())
                test_losses.append(test_loss.item())
                # Update tracking variables
                preds = output.detach().cpu().numpy()
                new_preds = np.zeros(preds.shape)
                for i in range(len(preds)):
                    new_preds[i][np.argmax(preds[i])] = 1
                test_accuracy = accuracy_score(y_true=batch['labels'].cpu().numpy().astype(int),
                                              y_pred=new_preds.astype(int))
                test_acc.append(test_accuracy)
                for i in range(len(preds)):
                    result_list = [e for e in preds[i]]
                    label_list = [e for e in labels.cpu().numpy()[i]]
                    final_pred_labels.append(result_list)
                    final_real_labels.append(label_list)

            test_loss_av = np.mean(test_losses)
            test_acc_av = np.mean(test_acc)
            print(f'Test Accuracy: {test_acc_av} Test Loss: {test_loss_av}')
            test['pred_labels'] = final_pred_labels
            test['real_labels'] = final_real_labels

    else:

        model = BERT_PLUS_MLP(bert, OUTPUT_DIM, 500)
        model.load_state_dict(torch.load('model_nn.pt', map_location=device))
        model.to(device)

        test_losses = []
        test_acc = []
        final_pred_labels = []
        final_real_labels = []
        model.eval()
        for batch in test_loader:
            with torch.no_grad():
                inputs, labels, attention_mask = batch['input_ids'].to(device), batch['labels'].to(device), batch[
                    'attention_mask'].to(device)
                outputs = model(input_ids=inputs.to(device),
                                attention_mask=attention_mask.to(device))
                criterion = torch.nn.BCELoss()
                loss = criterion(outputs.view(-1, OUTPUT_DIM),
                                 batch['labels'].type_as(outputs).view(-1, OUTPUT_DIM))
                test_losses.append(loss.item())
                # Update tracking variables
                preds = outputs.detach().cpu().numpy()
                new_preds = np.zeros(preds.shape)
                for i in range(len(preds)):
                    new_preds[i][np.argmax(preds[i])] = 1
                val_accuracy = accuracy_score(y_true=batch['labels'].cpu().numpy().astype(int),
                                              y_pred=new_preds.astype(int))
                test_acc.append(val_accuracy)
                for i in range(len(preds)):
                    result_list = [e for e in preds[i]]
                    label_list = [e for e in labels.cpu().numpy()[i]]
                    final_pred_labels.append(result_list)
                    final_real_labels.append(label_list)

        test_loss_av = np.mean(test_losses)
        test_acc_av = np.mean(test_acc)
        print(f'Test Accuracy: {test_acc_av} Test Loss: {test_loss_av}')
        test['pred_labels'] = final_pred_labels
        test['real_labels'] = final_real_labels

        if export_data is True:
            test_df.to_csv('test_predictions.csv')



# %% -------------------------------------- Model Classes ------------------------------------------------------------------
class BERT_PLUS_RNN(nn.Module):

    def __init__(self, bert, no_layers, hidden_dim, output_dim, batch_size):
        super(BERT_PLUS_RNN, self).__init__()
        self.bert = bert
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.embedding_dim = bert.config.to_dict()['hidden_size']
        self.no_layers = no_layers
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=no_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden, attention_mask):
        batch_size = x.size(0)
        embedded = self.bert(input_ids=x,attention_mask=attention_mask)[0]
        input = embedded.permute(1, 0, 2)
        #lstm_out, hidden = self.lstm(input, hidden)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        #out = self.fc(lstm_out)
        out = self.softmax(out)
        out = out.view(batch_size, -1, self.output_dim)
        out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        hidden = (h0, c0)
        return hidden

class BERT_PLUS_MLP(nn.Module):

    def __init__(self, bert, num_class, hidden_dim):
        super(BERT_PLUS_MLP, self).__init__()
        self.bert = bert
        self.fc1 = nn.Linear(bert.pooler.dense.out_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _, cls_hs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# step 1: load data from .csv from google drive
PATH = os.getcwd()
DATA_PATH = PATH + os.path.sep + 'Data'

# os.chdir(PATH + '/archive(4)/')
#
df = pd.read_csv(f'{DATA_PATH}/Tweets_test.csv')

# get data with only text and labels
test = df.copy()
print(f'shape of test data is {test.shape}')

test_loader = create_data_loader(test, tokenizer=tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)

# %% -------------------------------------- Model ------------------------------------------------------------------
bert = AutoModel.from_pretrained(checkpoint)
#freeze the pretrained layers
for param in bert.parameters():
    param.requires_grad = False

# run training loop, save model
Tester(model_type=model_type)

print('Done')


