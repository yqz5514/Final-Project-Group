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
import gdown
import matplotlib.pyplot as plt
import argparse

# %% -------------------------------------- Global Vars ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
OUTPUT_DIM = 3
BATCH_SIZE = 64
#MAX_VOCAB_SIZE = 25_000
MAX_LEN = 300
N_EPOCHS = 20
LR = 0.001
no_layers = 3
hidden_dim = 256
clip = 5
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model_type = 'MLP'
SAVE_MODEL = True #True to save the best model
print_metrics = True #True to plot train and val loss/accuracy

# %% -------------------------------------- Helper Functions ------------------------------------------------------------------
# remove contradictions
contraction_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                    "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
                    "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                    "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                    "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would",
                    "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
                    "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                    "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                    "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                    "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
                    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                    "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
                    "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                    "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                    "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",
                    "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
                    "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                    "there'd": "there would", "there'd've": "there would have", "there's": "there is",
                    "here's": "here is", "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                    "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                    "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                    "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
                    "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is",
                    "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
                    "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                    "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                    "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
                    "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                    "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
                    "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
                    "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}


def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re


contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]

    return contractions_re.sub(replace, text)
#remove stopwords
# def remove_stopwords(text):
#     """
#     Takes in a string of text, then performs the following:
#     1. Remove all stopwords
#     2. Returns a list of the cleaned text
#     """
#     STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
   
#     # Now remove any stopwords
#     return ' '.join([word for word in text.split() if word.lower() not in STOPWORDS])

def TextCleaning(text):
    '''
    Takes a string of text and performs some basic cleaning.
    1. removes tabs
    2. removes newlines
    3. removes special chars
    4. replaces contractions
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

    # Step 4: replace contractions 

    # Step 4
    s = replace_contractions(s)

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

def Trainer(model_type=model_type):
    if model_type == 'RNN':
        model = BERT_PLUS_RNN(bert, no_layers, hidden_dim, OUTPUT_DIM, BATCH_SIZE)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=LR)
        criterion = torch.nn.BCELoss()

        # Store our loss and accuracy for plotting

        epoch_tr_loss, epoch_vl_loss = [], []
        epoch_tr_acc, epoch_vl_acc = [], []
        met_test_best = 0

        for epoch in range(N_EPOCHS):
            # Tracking variables
            train_losses = []
            train_acc = []
            model.train()
            h = model.init_hidden(BATCH_SIZE)
            for batch in train_loader:
                inputs, labels, attention_mask = batch['input_ids'].to(device), batch['labels'].to(device), batch[
                    'attention_mask'].to(device)
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
                with torch.no_grad():
                    inputs, labels, attention_mask = batch['input_ids'].to(device), batch['labels'].to(device), batch[
                        'attention_mask'].to(device)
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

            # Sets the prioritized metric to be the validation accuracy
            met_test = epoch_val_acc

            # Saves the best model (assuming SAVE_MODEL=True at start): Code based on Exam 2 model saving code
            if met_test > met_test_best and SAVE_MODEL:
                torch.save(model.state_dict(), "model_onehot.pt")
                print("The model has been saved!")
                met_test_best = met_test
        if print_metrics is True:
            # Plots test vs train accuracy by epoch number
            plt.plot(range(epoch + 1), epoch_tr_acc, label="Train")
            plt.plot(range(epoch + 1), epoch_vl_acc, label="Val")
            plt.legend()
            plt.show()
            plt.savefig('accuracy_fig.png', bbox_inches='tight')

            # Clears plot so loss doesn't also show accuracy
            plt.clf()

            # Plots test vs train loss by epoch number
            plt.plot(range(epoch + 1), epoch_tr_loss, label="Train")
            plt.plot(range(epoch + 1), epoch_vl_loss, label="Val")
            plt.legend()
            plt.show()
            plt.savefig('loss_fig.png', bbox_inches='tight')

    else:

        # Store our loss and accuracy for plotting
        train_loss_set = []

        epoch_tr_loss, epoch_vl_loss = [], []
        epoch_tr_acc, epoch_vl_acc = [], []
        met_test_best = 0

        model = BERT_PLUS_MLP(bert, OUTPUT_DIM, 500)
        optimizer = AdamW(model.parameters(), lr=LR)

        num_training_steps = N_EPOCHS * len(train_loader)
        lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                                     num_warmup_steps=0, num_training_steps=num_training_steps)

        model.to(device)

        for epoch in range(N_EPOCHS):
            # Tracking variables
            train_losses = []
            train_acc = []
            model.train()
            for batch in train_loader:
                # clear previously calculated gradients
                model.zero_grad()
                outputs = model(input_ids=batch['input_ids'].to(device),
                                attention_mask=batch['attention_mask'].to(device))
                criterion = torch.nn.BCELoss()
                loss = criterion(outputs.view(-1, OUTPUT_DIM), batch['labels'].type_as(outputs).view(-1, OUTPUT_DIM))
                loss.backward()
                train_losses.append(loss.item())
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # Update tracking variables
                preds = outputs.detach().cpu().numpy()
                new_preds = np.zeros(preds.shape)
                for i in range(len(preds)):
                    new_preds[i][np.argmax(preds[i])] = 1
                accuracy = accuracy_score(y_true=batch['labels'].cpu().numpy().astype(int),
                                          y_pred=new_preds.astype(int))
                train_acc.append(accuracy)

            val_losses = []
            val_acc = []
            model.eval()
            for batch in valid_loader:
                with torch.no_grad():
                    outputs = model(input_ids=batch['input_ids'].to(device),
                                    attention_mask=batch['attention_mask'].to(device))
                    criterion = torch.nn.BCELoss()
                    loss = criterion(outputs.view(-1, OUTPUT_DIM),
                                     batch['labels'].type_as(outputs).view(-1, OUTPUT_DIM))
                    val_losses.append(loss.item())
                    # Update tracking variables
                    preds = outputs.detach().cpu().numpy()
                    new_preds = np.zeros(preds.shape)
                    for i in range(len(preds)):
                        new_preds[i][np.argmax(preds[i])] = 1
                    val_accuracy = accuracy_score(y_true=batch['labels'].cpu().numpy().astype(int),
                                                  y_pred=new_preds.astype(int))
                    val_acc.append(val_accuracy)

            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = np.mean(val_losses)
            epoch_train_acc = np.mean(train_acc)
            epoch_val_acc = np.mean(val_acc)
            epoch_tr_loss.append(epoch_train_loss)
            epoch_vl_loss.append(epoch_val_loss)
            epoch_tr_acc.append(epoch_train_acc)
            epoch_vl_acc.append(epoch_val_acc)
            print(f'Epoch {epoch + 1}')
            print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
            print(f'train_accuracy : {epoch_train_acc} val_accuracy : {epoch_val_acc}')

            # Sets the prioritized metric to be the validation accuracy
            met_test = epoch_val_acc

            # Saves the best model (assuming SAVE_MODEL=True at start): Code based on Exam 2 model saving code
            if met_test > met_test_best and SAVE_MODEL:
                torch.save(model.state_dict(), "model_onehot.pt")
                print("The model has been saved!")
                met_test_best = met_test
        if print_metrics is True:
            # Plots test vs train accuracy by epoch number
            plt.plot(range(epoch + 1), epoch_tr_acc, label="Train")
            plt.plot(range(epoch + 1), epoch_vl_acc, label="Val")
            plt.legend()
            plt.show()
            plt.savefig('accuracy_fig.png', bbox_inches='tight')

            # Clears plot so loss doesn't also show accuracy
            plt.clf()

            # Plots test vs train loss by epoch number
            plt.plot(range(epoch + 1), epoch_tr_loss, label="Train")
            plt.plot(range(epoch + 1), epoch_vl_loss, label="Val")
            plt.legend()
            plt.show()
            plt.savefig('loss_fig.png', bbox_inches='tight')


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
# step 1: load data from .csv from google drive. NOTE: need to fix, doesn't work downloaded from google.
parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True)  # Path of file
args = parser.parse_args()
PATH = args.path
# PATH = '/home/ubuntu/Final-Project-Group' #NOTE: need to change to arg parse
url = 'https://drive.google.com/uc?id=1YXhGD6NJ7mzYG78U9OgKnCq9pjM_u9zg&export=download'
DATA_PATH = PATH + os.path.sep + 'Data'
os.chdir(DATA_PATH)
gdown.download(url, 'Tweets.csv', quiet=False)


df = pd.read_csv(f'{DATA_PATH}/Tweets.csv')

# get data with only text and labels
df_copy = df.copy()
input_col = 'text'
label_col = 'airline_sentiment'
df_copy = df_copy[[input_col, label_col]]
df_copy = getLabel(df_copy, label_col, input_col)

# clean X data
df_copy[input_col] = df_copy[input_col].apply(TextCleaning)
#df_copy[input_col] = df_copy[input_col].apply(replace_contractions)

# split data
train, test = train_test_split(df_copy, train_size=0.8, random_state=SEED, stratify=df_copy['target'])
train, val = train_test_split(train, train_size=0.8, random_state=SEED, stratify=train['target'])

# save test data to use in test script
os.chdir(DATA_PATH)
test.to_csv("Tweets_test.csv")

print(f'shape of train data is {train.shape}')
print(f'shape of validation data is {val.shape}')
print(f'shape of test data is {test.shape}')

train_loader = create_data_loader(train, tokenizer=tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)
valid_loader = create_data_loader(val, tokenizer=tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)

# %% -------------------------------------- Model ------------------------------------------------------------------
bert = AutoModel.from_pretrained(checkpoint)
#freeze the pretrained layers
for param in bert.parameters():
    param.requires_grad = False

# run training loop, save model
Trainer(model_type=model_type)

print('Done')


