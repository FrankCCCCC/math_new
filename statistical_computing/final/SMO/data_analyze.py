# %%
from operator import index
import numpy as np
from numpy.core import numeric
import pandas as pd
from IPython.display import display

import re
from bs4 import BeautifulSoup
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from wordcloud import WordCloud
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

def read_original():
    return pd.read_csv('Womens Clothing E-Commerce Reviews.csv')

def prep_save():
    df = read_original()
    # Remove Null values
    df.drop(['Unnamed: 0', 'Title'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df.to_pickle("no_null.pkl")

    # Stemming & Remove stop words
    df['Review Text'] = df['Review Text'].apply(preprocess)

    print(df.isnull().sum())
    df.head()
    df.to_pickle("clean.pkl")

    return df
# %%
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def vec(review_text):
    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, norm='l2')
    count = CountVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

    # vec_text = count.fit_transform(review_text)
    vec_text = np.array(tfidf.fit_transform(review_text).toarray())

    return vec_text, tfidf

def lda(vec_text, n_comp):
    lda = LatentDirichletAllocation(n_components = n_comp, random_state = 0)

    clustered_text = lda.fit_transform(vec_text)

    pd.DataFrame(clustered_text).head()

    return clustered_text, lda

def vec_lda(review_text, n_comp):
    # n_comp = 5
    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, norm='l2')
    count = CountVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
    lda = LatentDirichletAllocation(n_components = n_comp, random_state = 0)

    # vec_text = np.array(count.fit_transform(review_text).toarray())
    vec_text = np.array(tfidf.fit_transform(review_text).toarray())
    clustered_text = np.array(lda.fit_transform(vec_text))

    pd.DataFrame(clustered_text).head()

    return clustered_text, vec_text, tfidf, lda
# %%
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
def plot_top_words(model, tfidf, n_top_words, title):
    feature_names = tfidf.get_feature_names()
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
import seaborn as sns
def plot_dist(df):
    # Age
    sns.distplot(x=df['Age'], bins=50, color='#9966ff', kde_kws={'bw':0.5})
    plt.title('Age Distribution', size=15)
    plt.xlabel('Age')
    plt.show()

    # Rating
    sns.distplot(x=df['Rating'], bins=5, color='#3399ff', kde=False, norm_hist=True)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title('Rating Distribution', size=15)
    plt.xticks(range(1, 6))
    plt.xlabel('Rating')
    plt.show()
# %%
# Read Dataset
is_read_original = False
if is_read_original:
    df = prep_save()
else:
    df = pd.read_pickle("clean.pkl")
    df_original = pd.read_pickle("no_null.pkl")
# %%
# Condition slicing
# df = df[df["Recommended IND"] == 1]
# df = df[df["Rating"] == 1]

# df = df[df["Department Name"] == "Intimate"]

# df = df[df["Class Name"] == "Blouses"]
# df = df[df["Class Name"] == "Bottoms"]

# df = df[df["Age"] >= 60]
# df = df[df["Age"] < 60]

df.head()

# %%
# Vectorization-LDA
vec_text, vectorizer = vec(df['Review Text'])
# clustered_text, lda = vec_lda(vec_text, n_comp=5)
# clustered_text, vec_text, vectorizer, lda = vec_lda(df['Review Text'], n_comp=5)

# %%
# Plot distribution of properties
# plot_dist(df)

# Average age & rating of each group
# groups = ['Division Name', 'Department Name', 'Class Name']
# fields = ['Age', 'Rating']

# for group in groups:
#     for field in fields:
#         print("Group By: ", group)
#         display(pd.DataFrame(df[[group, field]]).groupby(group).mean())

# %%
# Vectorizer features
# summary_text = np.sum(vec_text, axis=0)
# vectorizer._validate_vocabulary()
# display(pd.DataFrame(summary_text, index=vectorizer.get_feature_names()).nlargest(10, columns=[0]))

# Top words of LDA topics
# n_top_words = 10
# plot_top_words(lda, tfidf, n_top_words, 'Topics in LDA model')

# Word cloud
# cloud = WordCloud().generate(" ".join(list(df['Review Text'])))
# cloud.to_file('output.png')

# %%

# df_lda = pd.DataFrame(clustered_text)
# display(df_lda)
# df_txt = pd.DataFrame(df_original['Review Text'].iloc[np.argmax(clustered_text, axis=1)== 1])
# df_txt.to_csv("topic2_reviews.csv")
# display(df_txt)

# %%
# Dimension Reduction
from sklearn.decomposition import PCA
n_comp = 5
pca = PCA(n_components=n_comp)
pca.fit(vec_text)
X_pca = pca.transform(vec_text)
# %%
def label_proc(labels):
    labels[labels == 0] = -1
    return labels
# %%
from sklearn.model_selection import train_test_split
from svm import SVM as SVM, acc, draw_boundary

svm = SVM(C=0.6, max_iter=1000, kernel_type="rbf", gamma=2, epsilon=15)
X, y = X_pca, label_proc(df["Recommended IND"].to_numpy())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

svm.fit(X_train, y_train)

train_pred, test_pred, train_acc, test_acc = acc(X_train, X_test, y_train, y_test)

display(test_pred[:10])

# draw_boundary(X_train, y_train, svm.alpha, svm.b)
draw_boundary([(X, y)], [svm], ["RBF SVM"])

# %%
