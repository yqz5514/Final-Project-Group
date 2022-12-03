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
MAX_VOCAB_SIZE = 25_000
MAX_LEN = 300
N_EPOCHS = 10
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
    return DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=data_collator)


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
class BERT_PLUS_MLP(nn.Module):

    def __init__(self, bert, num_class, hidden_dim):
        super(BERT_PLUS_MLP, self).__init__()
        self.bert = bert
        self.fc1 = nn.Linear(bert.pooler.dense.out_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask):
        _, cls_hs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

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
model = BERT_PLUS_MLP(bert, OUTPUT_DIM, 500)


optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = N_EPOCHS * len(train_loader)
lr_scheduler = get_scheduler( "linear", optimizer=optimizer,
                              num_warmup_steps=0, num_training_steps=num_training_steps)
print(num_training_steps)

model.to(device)

# Store our loss and accuracy for plotting
train_loss_set = []

valid_loss_min = np.Inf
epoch_tr_loss, epoch_vl_loss = [], []
epoch_tr_acc, epoch_vl_acc = [], []

for epoch in range(N_EPOCHS):
    # Tracking variables
    train_losses = []
    train_acc = []
    model.train()
    for batch in train_loader:
        # clear previously calculated gradients
        model.zero_grad()
        outputs = model(input_ids= batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(outputs.view(-1, OUTPUT_DIM), batch['labels'].type_as(outputs).view(-1, OUTPUT_DIM))
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        # Update tracking variables
        THRESHOLD = 0.5
        preds = outputs.detach().cpu().numpy()
        preds[preds >= THRESHOLD] = 1
        preds[preds < THRESHOLD] = 0
        # accuracy = accuracy_score(y_true=batch['labels'].detach().cpu().numpy().astype(int),
        #                           y_pred=(np.rint(outputs.detach().cpu().numpy())).astype(int))
        accuracy = accuracy_score(y_true=batch['labels'].detach().cpu().numpy().astype(int),
                                  y_pred=preds.astype(int))
        train_acc.append(accuracy)

    val_losses = []
    val_acc = []
    model.eval()
    for batch in valid_loader:
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(outputs.view(-1, OUTPUT_DIM), batch['labels'].type_as(outputs).view(-1, OUTPUT_DIM))
            val_losses.append(loss.item())
            # Update tracking variables
            THRESHOLD = 0.5
            preds = outputs.detach().cpu().numpy()
            preds[preds >= THRESHOLD] = 1
            preds[preds < THRESHOLD] = 0
            # accuracy = accuracy_score(y_true=batch['labels'].detach().cpu().numpy().astype(int),
            #                           y_pred=(np.rint(outputs.detach().cpu().numpy())).astype(int))
            val_accuracy = accuracy_score(y_true=batch['labels'].detach().cpu().numpy().astype(int),
                                      y_pred=preds.astype(int))
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
test_losses = []
test_acc = 0.0
model.eval()
for batch in test_loader:
    outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(outputs.view(-1, OUTPUT_DIM), batch['labels'].type_as(outputs).view(-1, OUTPUT_DIM))
    test_losses.append(loss.item())
    # Update tracking variables
    THRESHOLD = 0.5
    preds = outputs.detach().cpu().numpy()
    preds[preds >= THRESHOLD] = 1
    preds[preds < THRESHOLD] = 0
    # accuracy = accuracy_score(y_true=batch['labels'].detach().cpu().numpy().astype(int),
    #                           y_pred=(np.rint(outputs.detach().cpu().numpy())).astype(int))
    accuracy = accuracy_score(y_true=batch['labels'].detach().cpu().numpy().astype(int),
                              y_pred=preds.astype(int))
    test_acc += accuracy
test_acc_av = test_acc/len(test_loader.dataset)
print(test_acc_av)