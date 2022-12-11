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

test_loader = create_data_loader(test, tokenizer=tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)

for param in bert.parameters():
    param.requires_grad = False

# run training loop, save model

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
# %% -------------------------------------- Data Prep -----------------------------------------------------------------

# %% -------------------------------------- Model ------------------------------------------------------------------
#freeze the pretrained layers
for param in bert.parameters():
    param.requires_grad = False

# %% -------------------------------------- Model Classes ------------------------------------------------------------------
class BERT_PLUS_RNN(nn.Module):

    def __init__(self, bert, no_layers, hidden_dim, output_dim, batch_size):
        super(BERT_PLUS_RNN, self).__init__()
        self.bert = bert
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.embedding_dim = bert.config.to_dict()['hidden_size']
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=no_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, hidden, attention_mask):
        batch_size = x.size(0)
        embedded = self.bert(input_ids=x,attention_mask=attention_mask)[0]
        input = embedded.permute(1, 0, 2)
        out = self.fc(out)
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


    def forward(self, input_ids, attention_mask):
        _, cls_hs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.fc2(x)
        return x

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
# ref tokens
ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence

visual = viz.visualize_text(vis_data_records)