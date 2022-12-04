# %% -------------------------------------- Imports ------------------------------------------------------------------
import os
import torch
import pandas as pd
import regex as re
from sklearn.model_selection import train_test_split
import torch.nn as nn
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoModel
from transformers import AdamW
from transformers import get_scheduler
from sklearn.metrics import accuracy_score
import numpy as np

# %% -------------------------------------- Global Vars ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
OUTPUT_DIM = 3
BATCH_SIZE = 64
#MAX_VOCAB_SIZE = 25_000
MAX_LEN = 300
N_EPOCHS = 5
LR = 0.001
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

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



# %% -------------------------------------- Model Class ------------------------------------------------------------------
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


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# step 1: load data from .csv
PATH = os.getcwd()
os.chdir(PATH + '/archive(4)/')

df = pd.read_csv('Tweets.csv')

# get data with only text and labels
df_copy = df.copy()
input_col = 'text'
label_col = 'airline_sentiment'
df_copy = df_copy[[input_col, label_col]]
df_copy = getLabel(df_copy, label_col, input_col)

# clean X data
df_copy[input_col] = df_copy[input_col].apply(TextCleaning)

# split data
train, test = train_test_split(df_copy, train_size=0.8, random_state=SEED, stratify=df_copy['target'])
train, val = train_test_split(train, train_size=0.8, random_state=SEED, stratify=train['target'])

print(f'shape of train data is {train.shape}')
print(f'shape of validation data is {val.shape}')
print(f'shape of test data is {test.shape}')

train_loader = create_data_loader(train, tokenizer=tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)
valid_loader = create_data_loader(val, tokenizer=tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)
test_loader = create_data_loader(test, tokenizer=tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)

# %% -------------------------------------- Model ------------------------------------------------------------------
bert = AutoModel.from_pretrained(checkpoint)
#freeze the pretrained layers
for param in bert.parameters():
    param.requires_grad = False
no_layers = 3
hidden_dim = 256
clip = 5
model = BERT_PLUS_RNN(bert, no_layers, hidden_dim, OUTPUT_DIM, BATCH_SIZE)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)
criterion = torch.nn.BCELoss()

# Store our loss and accuracy for plotting

valid_loss_min = np.Inf
epoch_tr_loss, epoch_vl_loss = [], []
epoch_tr_acc, epoch_vl_acc = [], []



for epoch in range(N_EPOCHS):
    # Tracking variables
    train_losses = []
    train_acc = []
    model.train()
    h = model.init_hidden(BATCH_SIZE)
    for batch in train_loader:
        inputs, labels, attention_mask = batch['input_ids'].to(device), batch['labels'].to(device), batch['attention_mask'].to(device)
        h = tuple([each.data for each in h])
        model.zero_grad()
        output, h = model(inputs, h, attention_mask)
        loss = criterion(output, labels.float())
        loss.backward()
        train_losses.append(loss.item())
        # Update tracking variables
        preds = output.detach().cpu().numpy()
        new_preds = np.zeros(preds.shape)
        for i in range(len(preds)):
            new_preds[i][np.argmax(preds[i])] = 1
        accuracy = accuracy_score(y_true=batch['labels'].cpu().numpy().astype(int),
                                  y_pred=new_preds.astype(int))
        train_acc.append(accuracy)
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    val_losses = []
    val_acc = []
    model.eval()
    val_h = model.init_hidden(BATCH_SIZE)
    for batch in valid_loader:
        inputs, labels, attention_mask = batch['input_ids'].to(device), batch['labels'].to(device), batch['attention_mask'].to(device)
        val_h = tuple([each.data for each in val_h])
        output, val_h = model(inputs, val_h, attention_mask)
        val_loss = criterion(output, labels.float())
        val_losses.append(val_loss.item())
        # Update tracking variables
        preds = output.detach().cpu().numpy()
        new_preds = np.zeros(preds.shape)
        for i in range(len(preds)):
            new_preds[i][np.argmax(preds[i])] = 1
        val_accuracy = accuracy_score(y_true=batch['labels'].cpu().numpy().astype(int),
                                  y_pred=new_preds.astype(int))
        val_acc.append(val_accuracy)

    epoch_train_loss = np.mean(train_losses)
    epoch_val_loss = np.mean(val_losses)
    # epoch_train_acc = train_acc / len(train_loader.dataset)
    # epoch_val_acc = val_acc / len(valid_loader.dataset)
    epoch_train_acc = np.mean(train_acc)
    epoch_val_acc = np.mean(val_acc)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_vl_loss.append(epoch_val_loss)
    epoch_tr_acc.append(epoch_train_acc)
    epoch_vl_acc.append(epoch_val_acc)
    print(f'Epoch {epoch + 1}')
    print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    print(f'train_accuracy : {epoch_train_acc} val_accuracy : {epoch_val_acc}')

# Test Set:
# test_losses = []
# test_acc = []
# model.eval()
# test_h = model.init_hidden(BATCH_SIZE)
# for batch in valid_loader:
#     inputs, labels, attention_mask = batch['input_ids'].to(device), batch['labels'].to(device), batch['attention_mask'].to(device)
#     test_h = tuple([each.data for each in test_h])
#     output, test_h = model(inputs, test_h, attention_mask)
#     test_loss = criterion(output, labels.float())
#     test_losses.append(test_loss.item())
#     # Update tracking variables
#     preds = output.detach().cpu().numpy()
#     new_preds = np.zeros(preds.shape)
#     for i in range(len(preds)):
#         new_preds[i][np.argmax(preds[i])] = 1
#     test_accuracy = accuracy_score(y_true=batch['labels'].cpu().numpy().astype(int),
#                               y_pred=new_preds.astype(int))
#     test_acc.append(test_accuracy)
# test_acc_av = test_acc/len(test_loader.dataset)
# print(test_acc_av)


# %% -------------------------------------- Interpretation ------------------------------------------------------------------
import IPython
import shap
from IPython.core.display import HTML

batch = next(iter(train_loader))
inputs, labels, attention_mask = batch['input_ids'], batch['labels'], batch['attention_mask']
batch_hidden = model.init_hidden(BATCH_SIZE)
batch_hidden = tuple([each.data for each in batch_hidden])


#
explainer = shap.Explainer(model, [inputs, batch_hidden, attention_mask] )
#
shap_values = explainer([inputs, batch_hidden, attention_mask])

shap.plots.text(shap_values[2])

IPython.core.display.HTML(data=shap.plots.text(shap_values))