#%%-----------------------------------------dataprocessing---------------------------------------------------------------
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

##%%-----------------------------------------------------------Lime----------------------------------------------------
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
    preds = output.detach().cpu().numpy().reshape(1, -1)

    # for batch in test_loader:
    #     outputs = model(input_ids=batch[input_ids].to(device), attention_mask=batch['attention_mask'].to(device))
    #     preds = outputs.detach().cpu().numpy()
    #     return preds
    # inputs = [z['input_ids'], z['attention_mask']]
    # k = []
    # k.append(float(output.reshape(-1, 1)))
    # #k.append(float(1 - output.reshape(-1, 1)))
    # k = np.array(k).reshape(1, -1)

    return preds

#input_ids = '13789'
STR = str(test.text[13789])
exp = explainer.explain_instance(STR, predict_probab, num_features=10, num_samples=1)
#%%
df2 = df[['text','airline_sentiment']]
#%%
df2['text'] = df2['text'].apply(text_process)
#%%
# s_mapping = {
#            'neutral': 1,
#            'positive': 2,
#             'negative': 3}
# df2['airline_sentiment'] = df2['airline_sentiment'].map(s_mapping)

#%%----------------------------------------lime

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

  
