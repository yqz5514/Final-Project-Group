
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

def create_data_loader(df, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    ds = nlpDataset(
        data_frame = df,
        tokenizer = tokenizer,
        max_len = max_len,
        input_col = input_col)
    return DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=data_collator, drop_last=False)

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
        model.load_state_dict(torch.load('model_onehot.pt', map_location=device))
        model.to(device)

        criterion = torch.nn.BCELoss()

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
        model.load_state_dict(torch.load('model_onehot.pt', map_location=device))
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
            test.to_csv('test_predictions.csv')


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

# step 1: load data from .csv 
parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True)  # Path of file
args = parser.parse_args()
PATH = args.path
#PATH = 'home/ubuntu/Final-Project-Group'
DATA_PATH = PATH + os.path.sep + 'Data'
MODEL_PATH = PATH + os.path.sep + 'Data'
#
df = pd.read_csv(f'{DATA_PATH}/Tweets_test.csv')

os.chdir(DATA_PATH)

# get data with only text and labels
test = df.copy()
print(f'shape of test data is {test.shape}')
input_col = 'text'
label_col = 'target'
def cleanLabel(row):
    '''
    cleans function after loading from csv file
    :param row:
    :return:
    '''
    remove = ['[', ']', ',', ' ']
    label = [float(i) for i in row if i not in remove]
    return label

test[label_col] = test[label_col].apply(cleanLabel)

test_loader = create_data_loader(test, tokenizer=tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)

bert = AutoModel.from_pretrained(checkpoint)
#freeze the pretrained layers
for param in bert.parameters():
    param.requires_grad = False

# run training loop, save model
Tester(model_type=model_type)

print('Done')

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
            plt.plt(range(epoch + 1), epoch_tr_loss, label="Train")
            plt.plt(range(epoch + 1), epoch_vl_loss, label="Val")
            plt.legend()
            plt.show()
            plt.savefig('loss_fig.png', bbox_inches='tight')
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
PATH = '/home/ubuntu/Final-Project-Group'
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

    # Downloads the model from Google Drive
# Run
parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True)  # Path of file
args = parser.parse_args()
PATH = args.path
MODEL_PATH = PATH + os.path.sep + 'Data'

os.chdir(MODEL_PATH)
url = 'https://drive.google.com/u/3/uc?id=1q2QAYGM-hmFxmH_ytHW-L4lw_792iAEx&export=download'
gdown.download(url, 'model_onehot.pt', quiet=False)
print('Done!')