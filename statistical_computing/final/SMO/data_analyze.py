# %%
import numpy as np
import pandas as pd

import re
from bs4 import BeautifulSoup
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# %%
def remove_tags(text):
    # remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    
    # regex for matching emoticons, keep emoticons, ex: :), :-P, :-D
    r = '(?::|;|=|X)(?:-)?(?:\)|\(|D|P)'
    emoticons = re.findall(r, text)
    text = re.sub(r, '', text)
    
    # convert to lowercase and append all emoticons behind (with space in between)
    # replace('-','') removes nose of emoticons
    text = re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-','')
    return text

nltk.download('stopwords')
def tokenizer_stem_nostop(text):
    stop = stopwords.words('english')
    porter = PorterStemmer()
    return [porter.stem(w) for w in re.split('\s+', text.strip()) \
            if w not in stop and re.match('[a-zA-Z]+', w)]

def preprocess(text):
    text = remove_tags(text)
    text = tokenizer_stem_nostop(text)
    text = ' '.join(text)
    return text

def prep_save():
    df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv')
    # Remove Null values
    df.drop(['Unnamed: 0', 'Title'], axis=1, inplace=True)
    df.dropna(inplace=True)

    df['Review Text'] = df['Review Text'].apply(preprocess)

    print(df.isnull().sum())
    df.head()
    df.to_pickle("clean.pkl")

# %%
is_read_original = False
if is_read_original:
    prep_save()
else:
    df = pd.read_pickle("clean.pkl")
# %%
# df = df[df["Recommended IND"] == 1]
# df = df[df["Rating"] == 1]

# df = df[df["Department Name"] == "Intimate"]

# df = df[df["Class Name"] == "Blouses"]
# df = df[df["Class Name"] == "Bottoms"]

df = df[df["Age"] >= 60]
# df = df[df["Age"] < 60]

df.head()

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline

n_comp = 5
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, norm='l2')
lda = LatentDirichletAllocation(n_components = n_comp, random_state = 0)

lda_pipe = Pipeline([('tfidf', tfidf), ('lda', lda)])

# %%
clustered_text = lda_pipe.fit_transform(df['Review Text'])
# %%
import matplotlib.pyplot as plt
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

# %%
pd.DataFrame(clustered_text).head()

n_top_words = 10
tf_feature_names = tfidf.get_feature_names()
plot_top_words(lda, tf_feature_names, n_top_words, 'Topics in LDA model')
# %%
from wordcloud import WordCloud

cloud = WordCloud().generate(" ".join(list(df['Review Text'])))
cloud.to_file('output.png')
# %%


# %%

# %%
