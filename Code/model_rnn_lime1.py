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
import string
from nltk.corpus import stopwords
#%%
#%%
os.chdir('/home/ubuntu/test/Final-Project-Group/Code')
os.getcwd()
#%%
df = pd.read_csv('Tweets.csv')
#%%
df.head()
#%%
df1 = df[['text','airline_sentiment']].copy()
df1['WORD_COUNT'] = df1['text'].apply(lambda x: len(x.split()))
#%%
df1.describe()
#  WORD_COUNT
# count  14640.000000
# mean      17.653415
# std        6.882259
# min        2.000000
# 25%       12.000000
# 50%       19.000000
# 75%       23.000000
# max       36.000000

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
LR = 0.0001
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
def text_process(text):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
    # Check characters to see if they are in punctuation
    nopunc = [char for char in text if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])


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

def getLabel(df, label_col, input_col):
    encoded = pd.get_dummies(df, columns=[label_col])
    encoded_val = encoded.iloc[:, 1:].apply(list, axis=1)
    encoded['target'] = encoded_val
    return_df = encoded[[input_col, 'target']]
    return return_df



def create_data_loader(df, tokenizer, max_len=MAX_LEN,batch_size=BATCH_SIZE):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    ds = TextDataset(
        data_frame = df,
        tokenizer = tokenizer,
        max_len = max_len,
        input_col = input_col)
    return DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=data_collator, drop_last=True)


# %% -------------------------------------- Dataset Class ------------------------------------------------------------------

class TextDataset(Dataset):
    def __init__(self, data_frame, tokenizer, max_len,input_col):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input = self.data_frame.iloc[idx][input_col]
        target = self.data_frame.iloc[idx]['target_list']
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
        #self.batch_size = 1
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
# PATH = os.getcwd()
# os.chdir(PATH + '/archive(4)/')
#%%
os.chdir('/home/ubuntu/test/Final-Project-Group/Code')
#%%
os.getcwd()
#%%
df = pd.read_csv("Tweets.csv")

# get data with only text and labels
df_copy = df.copy()
input_col = 'text'
label_col = 'airline_sentiment'
df_copy = df_copy[[input_col, label_col]]
df_copy = getLabel(df_copy, label_col, input_col)

# clean X data
df_copy[input_col] = df_copy[input_col].apply(lambda x: x.lower())
#df_copy[input_col] = df_copy[input_col].apply(replace_contractions)
df_copy[input_col] = df_copy[input_col].apply(text_process)
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
#%%-----------------------------------------------Lime_1----------------------------------------------------------
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=['neutral', 'positive', 'negative'])


def predict_probab(STR):
    z = tokenizer.encode_plus(
        STR,
        add_special_tokens=True,
        max_length=200,
        return_attention_mask=True,
        is_split_into_words=True,
        truncation=True,
        padding='max_length',
        return_tensors='pt', )
    # z = tokenizer.encode_plus(STR, add_special_tokens=True, max_length=512, truncation=True, padding='max_length',
    #                           return_token_type_ids=True, return_attention_mask=True, return_tensors='np')
    #inputs = [z['input_ids'], z['attention_mask']]
    inputs, attention_mask = z['input_ids'].to(device),z['attention_mask'].to(device)
    h = model.init_hidden(1)
    h = tuple([each.data for each in h])
    output, h = model(inputs, h, attention_mask)
    preds = output.detach().cpu().numpy()

    # for batch in test_loader:
    #     outputs = model(input_ids=batch[input_ids].to(device), attention_mask=batch['attention_mask'].to(device))
    #     preds = outputs.detach().cpu().numpy()
    #     return preds
    # inputs = [z['input_ids'], z['attention_mask']]
    # k = []
    # k.append(float(model.predict(inputs).reshape(-1, 1)))
    # k.append(float(1 - model.predict(inputs).reshape(-1, 1)))
    # k = np.array(k).reshape(1, -1)

    return preds

#input_ids = '13789'
STR = str(test.text[13789])
exp = explainer.explain_instance(STR, predict_probab, num_features=10, num_samples=1)

#%%----------------------------------------------LIME2----------------------------------------------
# used on dateset

# df2 = df[['text','airline_sentiment']]

# df2['text'] = df2['text'].apply(text_process)

class_names = ['positive','negative', 'neutral']

results = []
def predictor(STR):
    z = tokenizer.encode_plus(
            STR,
            add_special_tokens=True,
            max_length=300,
            return_attention_mask=True,
            is_split_into_words=True,
            truncation=True,
            padding=True,
            return_tensors='pt',
            )
        #output = model(**tokenizer(STR, return_tensors="pt", padding=True))

    inputs, attention_mask = z['input_ids'].to(device), z['attention_mask'].to(device)
    # for batch in test_loader:
    #     output = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))

    h = model.init_hidden(1)
    h = tuple([each.data for each in h])
    output,h = model(inputs, h, attention_mask)
    #probas = F.softmax(model(inputs, h, attention_mask).logits).detach().numpy()
    with torch.no_grad():
        #output = model(inputs, h, attention_mask)
           logits = output[0]

           logits = F.softmax(logits,dim=0)
           results.append(logits.cpu().detach().numpy()[0])
           results_array = np.array(results)

    return results_array

    #probas = F.softmax(h, dim = 1).cpu().detach().numpy()

    # results.append(h.detach()[0])
    #
    # ress = [res for res in results]
    # results_array = np.array(ress)
    # return results_array



# results.append(raw_outputs[0])
#
# ress = [res for res in results]
# results_array = np.array(ress)
# return results_array
#
# outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
# probas = F.softmax(outputs.logits).detach().numpy()
        #return probas
#%%
#predictor(str(test.text[13789]))
#%%
explainer = LimeTextExplainer(class_names=class_names)

STR = str(test.text[2998])
exp = explainer.explain_instance(STR, predictor, num_features=20, num_samples=2000)
exp.show_in_notebook(text=STR)
