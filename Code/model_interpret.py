# Adapted from https://captum.ai/tutorials/Bert_SQUAD_Interpret

# %% -------------------------------------- Imports ------------------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from transformers import AutoTokenizer, AutoModel
from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients
import argparse


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

# %% -------------------------------------- Global Vars ------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# replace <PATH-TO-SAVED-MODEL> with the real path of the saved model
model_type = 'MLP'
checkpoint = "bert-base-uncased"
model_type = 'MLP'

# %% -------------------------------------- Helper Functions ------------------------------------------------------------------
def DefineModel(model_type=model_type):
    OUTPUT_DIM = 3
    BATCH_SIZE = 64
    no_layers = 3
    hidden_dim = 256
    bert = AutoModel.from_pretrained(checkpoint)
    # freeze the pretrained layers
    for param in bert.parameters():
        param.requires_grad = False

    if model_type == 'RNN':
        model = BERT_PLUS_RNN(bert, no_layers, hidden_dim, OUTPUT_DIM, BATCH_SIZE)
        model.load_state_dict(torch.load('model_nn.pt', map_location=device))
        model.to(device)

    else:
        model = BERT_PLUS_MLP(bert, OUTPUT_DIM, 500)
        model.load_state_dict(torch.load('model_nn.pt', map_location=device))
        model.to(device)

    return model

def predict(inputs, attention_mask=None):
    output = model(input_ids=inputs, attention_mask=attention_mask)
    preds = output
    new_preds = torch.zeros(preds.shape)
    for i in range(len(preds)):
        new_preds[i][torch.argmax(preds[i])] = 1
    return output

def construct_input(text, cls_token_id, ref_token_id, sep_token_id):
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    baseline_ids = [cls_token_id] + [ref_token_id] * (len(input_ids)-2) + [sep_token_id]
    return torch.tensor([input_ids], device=device), torch.tensor([baseline_ids], device=device)

def construct_attention_mask(input_ids):
    mask = torch.ones_like(input_ids)
    baseline = torch.zeros_like(input_ids)
    return mask, baseline

def construct_whole_bert_embeddings(input_ids):
    input_embeddings = model.bert.embeddings(input_ids)
    return input_embeddings

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def TextInterpreter(ex):
    input_ids, baseline_ids = construct_input(ex[0], cls_token_id=cls_token_id, ref_token_id=ref_token_id, sep_token_id=sep_token_id)
    attention_mask, attention_baseline = construct_attention_mask(input_ids)

    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)

    predicted_output = predict(input_ids, attention_mask)
    preds = predicted_output[0]
    new_preds = torch.zeros(preds.shape)
    new_preds[torch.argmax(preds)] = 1
    prediction_score, pred_label_idx = torch.topk(preds, 1)
    pred_label_idx.squeeze()
    pred_label_int = pred_label_idx.squeeze().cpu().numpy().item()
    real_label_int = np.argmax(ex[1]).item()
    classes_dict = {0: 'negative', 1: 'neutral', 2: 'positive'}
    pred_label_string = classes_dict[pred_label_int]
    real_label_string = classes_dict[real_label_int]
    lig = LayerIntegratedGradients(predict, model.bert.embeddings)

    attributions_lig, delta = lig.attribute(inputs=input_ids,
                                       baselines=baseline_ids,
                                       target=pred_label_int,
                                       additional_forward_args=attention_mask,
                                       return_convergence_delta=True)

    attributions_sum = summarize_attributions(attributions_lig)

    summary_vis = viz.VisualizationDataRecord(
                            word_attributions=attributions_sum,
                            pred_prob=torch.max(preds),
                            pred_class=pred_label_string,
                            true_class=real_label_string,
                            attr_class=str(pred_label_int),
                            attr_score=attributions_sum.sum(),
                            raw_input_ids=all_tokens,
                            convergence_score=delta)
    vis_data_records.append(summary_vis)
    return

# %% -------------------------------------- Main: Captum  ------------------------------------------------------------------
# load model
parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True)  # Path of file
args = parser.parse_args()
PATH = args.path
MODEL_PATH = PATH + os.path.sep + 'Data'
DATA_PATH = PATH + os.path.sep + 'Data'
os.chdir(MODEL_PATH)
model = DefineModel(model_type=model_type)
model.eval()
model.zero_grad()

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# ref tokens
ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence

# load data
df = pd.read_csv(f'{DATA_PATH}/Tweets_test.csv')

# create examples, do basic preprocessing on labels
vis_data_records = []
examples = []
n=10
for i in range(n):
    text, label = df.iloc[i]['text'], df.iloc[i]['target']
    remove = ['[', ']', ',', ' ']
    label = [float(i) for i in label if i not in remove]
    examples.append([text, label])

for example in examples:
    TextInterpreter(example)

visual = viz.visualize_text(vis_data_records)
with open("data.html", "w") as file:
    file.write(visual.data)
