import os
import torch
import pandas as pd
import regex as re
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
import seaborn as sns
import argparse
import gdown
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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




# remove username handles
def remove_usernames_links(text):
    text = re.sub('@[^\s]+', '', str(text))
    text = re.sub('http[^\s]+', '', str(text))
    return text

df['text'] = df['text'].apply(remove_usernames_links)





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

def TextCleaning(text):

    pattern1 = re.compile(r'\<.*?\>')
    s = re.sub(pattern1, '', text)

    pattern2 = re.compile(r'\n')
    s = re.sub(pattern2, ' ', s)

    pattern3 = re.compile(r'[^0-9a-zA-Z!/?]+')
    s = re.sub(pattern3, ' ', s)

    pattern4 = re.compile(r"n't")
    s = re.sub(pattern4, " not", s)

    return s


df['text'] = df['text'].apply(TextCleaning)


def polarity(text):
    return TextBlob(text).sentiment.polarity

df['text_polarity']=df['text'].apply(lambda x : polarity(x))

# overall polatity
plt.figure(1)
pol_hist = df['text_polarity'].hist()
plt.xlabel("Polarity")
plt.ylabel("Data Count")
plt.title("Distribution of Polarity")
# plt.show()
plt.savefig("Distribution of Polarity",bbox_inches='tight')


# polatity for each airline
plt.figure(2)
sns.boxplot(x = 'airline', y = 'text_polarity', data = df)
plt.xlabel("Airlines")
plt.ylabel("Polatiry")
plt.title("Airline vs Polarity")
# plt.show()
plt.savefig("Airline vs Polarity",bbox_inches='tight')

# count per class
plt.figure(3)
df['airline_sentiment'].value_counts().plot(kind='bar')
plt.xlabel("Sentiment",labelpad=14)
plt.ylabel("Count of Tweets",labelpad=14)
plt.title("Count of Tweets per Sentiment")
plt.savefig("Count of Tweets per Sentiment",bbox_inches='tight')


stopwords = set(STOPWORDS)

#positive sentiments
plt.figure(4)
df_pos = df[df['airline_sentiment']=='positive']
pos_text = " ".join(str(i) for i in df_pos.text)
wordcloud_pos = WordCloud(stopwords=stopwords, background_color="black").generate(pos_text)
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.tick_params(left = False, bottom = False)
plt.xticks([]);plt.yticks([])
plt.title("Positive Sentiments")
# plt.show()
plt.savefig("Positive Sentiments",bbox_inches='tight')


#neutral sentiments
plt.figure(5)
df_net = df[df['airline_sentiment']=='neutral']
net_text = " ".join(str(i) for i in df_net.text)
wordcloud_net = WordCloud(stopwords=stopwords, background_color="black").generate(net_text)
plt.tick_params(left = False, bottom = False)
plt.xticks([]);plt.yticks([])
plt.imshow(wordcloud_net, interpolation='bilinear')
plt.title("Neutral Sentiments")
# plt.show()
plt.savefig("Neutral Sentiments",bbox_inches='tight')

#neutral sentiments
plt.figure(6)
df_neg = df[df['airline_sentiment']=='negative']
neg_text = " ".join(str(i) for i in df_neg.text)
wordcloud_neg = WordCloud(stopwords=stopwords, background_color="black").generate(neg_text)
plt.tick_params(left = False, bottom = False)
plt.xticks([]);plt.yticks([])
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.title("Negative Sentiments")
# plt.show()
plt.savefig("Negative Sentiments",bbox_inches='tight')


df["text Length"]= df["text"].str.len()

def show_dist(df, col):
    print('Descriptive stats for {}'.format(col))
    print('-'*(len(col)+22))
    print(df.groupby('airline_sentiment')[col].describe())
    bins = np.arange(df[col].min(), df[col].max() + 1)
    g = sns.FacetGrid(df, col='airline_sentiment', height=5, hue='airline_sentiment', palette="PuBuGn_d")
    g = g.map(sns.histplot, col, kde=False, bins=bins)
    plt.savefig("Length",bbox_inches='tight')

show_dist(df,"text Length")

#LIME
# from lime.lime_text import LimeTextExplainer
# #print(test)
# test_doc = test['text'].tolist()
#
# #print(test_doc.shape)
# class_names = ['positive', 'neutral', 'negative']
#
# def predictor(texts):
#     outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
#     tensor_logits = outputs[0].cpu()
#     probas = F.softmax(tensor_logits).detach().numpy()
#
#
#     return probas
#
# c=predictor(test_doc[0])
# print(c.shape)

# print(c)
# explainer = LimeTextExplainer(class_names=class_names)
# exp = explainer.explain_instance(test_doc[0], predictor, num_samples=1, labels = (2,))

# SHAP
# from torch.utils.data import DataLoader
# import transformers

# import IPython
# from IPython.core.display import HTML


# tokenizer = transformers.AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion", use_fast=True)
# model = transformers.AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion").cuda()

# pred = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer, device=0, return_all_scores=True)
# #
# explainer = shap.Explainer(pred)
# #
# shap_values = explainer(train[input_col][:3])

# shap.plots.text(shap_values[2])

# IPython.core.display.HTML(data=shap.plots.text(shap_values))
