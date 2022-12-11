
def create_data_loader(df, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    ds = nlpDataset(
        data_frame = df,
        tokenizer = tokenizer,
        max_len = max_len,
        input_col = input_col)
    return DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=data_collator, drop_last=False)

# %% -------------------------------------- Dataset Class ------------------------------------------------------------------
class nlpDataset(Dataset):
    def __init__(self, data_frame, tokenizer, max_len, input_col):
        self.data_frame = data_frame

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input = self.data_frame.iloc[idx][input_col]
        target = self.data_frame.iloc[idx]['target']

        output = {
            'labels': torch.tensor(target, dtype=torch.long)
        }

        return output

     self.bert = bert
        embedded = self.bert(input_ids=x,attention_mask=attention_mask)[0]

        self.bert = bert
bert = AutoModel.from_pretrained(checkpoint)

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