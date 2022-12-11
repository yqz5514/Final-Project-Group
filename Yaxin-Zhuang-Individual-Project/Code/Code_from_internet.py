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
#%%------------------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
class_names = ['positive','negative', 'neutral']

def predictor(texts):
    outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
    probas = F.softmax(outputs.logits).detach().numpy()
    return probas

explainer = LimeTextExplainer(class_names=class_names)

str_to_predict = "surprising increase in revenue in spite of decrease in market share"
exp = explainer.explain_instance(str_to_predict, predictor, num_features=20, num_samples=2000)
exp.show_in_notebook(text=str_to_predict)
#%%-----------------------------------------------------------------------------------
def predictor(texts):
    outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
    tensor_logits = outputs[0].cpu()
    probas = F.softmax(tensor_logits).detach().numpy()
    return probas
c=predictor(test_doc[0])
print(c.shape)
text="hello my bame is"
print(c)
explainer = LimeTextExplainer(class_names=class_names)
exp = explainer.explain_instance(test_doc[0], predictor, num_features=20, num_samples=2000)
class Prediction:

    def __init__(self, bert_model_class, model_path, lower_case, seq_length):

        self.model, self.tokenizer, self.model_config = \
                    self.load_model(bert_model_class, model_path, lower_case)
        self.max_seq_length = seq_length
        self.device = "cpu"
        self.model.to("cpu")

    def load_model(self, bert_model_class, model_path, lower_case):

        config_class, model_class, tokenizer_class = MODEL_CLASSES[bert_model_class]
        config = config_class.from_pretrained(model_path)
        tokenizer = tokenizer_class.from_pretrained(model_path, do_lower_case=lower_case)
        model = model_class.from_pretrained(model_path, config=config)

        return model, tokenizer, config

    def predict_label(self, text_a, text_b):

        self.model.to(self.device)

        input_ids, input_mask, segment_ids = self.convert_text_to_features(text_a, text_b)
        with torch.no_grad():
            outputs = self.model(input_ids, segment_ids, input_mask)

        logits = outputs[0]
        logits = F.softmax(logits, dim=1)
        # print(logits)
        logits_label = torch.argmax(logits, dim=1)
        label = logits_label.detach().cpu().numpy()

        # print("logits label ", logits_label)
        logits_confidence = logits[0][logits_label]
        label_confidence_ = logits_confidence.detach().cpu().numpy()
        # print("logits confidence ", label_confidence_)

        return label, label_confidence_


    def _truncate_seq_pair(self, tokens_a, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a)
            if total_length <= max_length:
                break
            if len(tokens_a) > max_length:
                tokens_a.pop()

    def convert_text_to_features(self, text_a, text_b=None):

        features = []
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        cls_token_at_end = False
        sequence_a_segment_id = 0
        sequence_b_segment_id = 1
        cls_token_segment_id = 1
        pad_token_segment_id = 0
        mask_padding_with_zero = True
        pad_token = 0
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens_b = None

        self._truncate_seq_pair(tokens_a, self.max_seq_length - 2)

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)


        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        #
        # # Zero-pad up to the sequence length.
        padding_length = self.max_seq_length - len(input_ids)


        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        input_mask = torch.tensor([input_mask], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long).to(self.device)

        return input_ids, input_mask, segment_ids

    def predictor(self, text):

        examples = []
        print(text)
        for example in text:
            examples.append(self.convert_text_to_features(example))

        results = []
        for example in examples:

            with torch.no_grad():
                outputs = self.model(example[0], example[1], example[2])
            logits = outputs[0]
            logits = F.softmax(logits, dim = 1)
            results.append(logits.cpu().detach().numpy()[0])

        results_array = np.array(results)

        return results_array
if __name__ == '__main__':

    model_path = "models/mrpc"
    bert_model_class = "bert"
    prediction = Prediction(bert_model_class, model_path,
                                lower_case = True, seq_length = 512)
    label_names = [0, 1]
    explainer = LimeTextExplainer(class_names=label_names)
    train_df = pd.read_csv("data/train.tsv", sep = '\t')

    train_ls = train_df["string"].tolist()

    for example in train_ls:

        exp = explainer.explain_instance(example, prediction.predictor)
        words = exp.as_list()
