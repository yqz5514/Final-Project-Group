#%%---------------------------------dataprocessing--------------------------------------------------------
os.chdir('/home/ubuntu/test/Final-Project-Group/Code')
os.getcwd()
df = pd.read_csv('Tweets.csv')
df.head()
df1 = df[['text','airline_sentiment']].copy()
df1['WORD_COUNT'] = df1['text'].apply(lambda x: len(x.split()))
df1.describe()
#remove stopwords
def remove_stopwords(text):
    """
    Takes in a string of text, then performs the following:
    1. Remove all stopwords
    2. Returns a list of the cleaned text
    """
    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
   
    # Now remove any stopwords
    return ' '.join([word for word in text.split() if word.lower() not in STOPWORDS])
# clean data
df_copy[input_col] = df_copy[input_col].apply(lambda x: x.lower())
df_copy[input_col] = df_copy[input_col].apply(replace_contractions)
#%%-----------------------------------Lime-experiment-----------------------------------------------------------------
class_names=['neutral', 'positive', 'negative']
def predictor(STR):
    z = tokenizer.encode_plus(
        STR,
        add_special_tokens=True,
        max_length=200,
        return_attention_mask=True,
        is_split_into_words=True,
        truncation=True,
        padding='max_length',
        return_tensors='pt', )
    inputs, attention_mask = z['input_ids'].to(device), z['attention_mask'].to(device)
    h = model.init_hidden(1)
    h = tuple([each.data for each in h])
    output, h = model(inputs, h, attention_mask)
    result = []
    logits = output[0]
    logits = F.softmax(logits).cpu().numpy()
    result.append(logits.detach().cpu().numpy()[0])
    results_array = np.array(result)
    return results_array
explainer = LimeTextExplainer(class_names=class_names)

STR = str(test.text[13789])
exp = explainer.explain_instance(STR, predictor, num_features=20, num_samples=2000)
exp.show_in_notebook(text=STR)
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
