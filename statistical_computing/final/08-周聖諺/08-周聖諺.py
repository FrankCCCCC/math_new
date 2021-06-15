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
#%%
try:
    import jax.numpy as snp
    print("Use Jax.numpy module")
except:
    import numpy as snp
    print("Didn't find out Jax module, use numpy instead")

import pickle
import sys
from enum import Enum
import random
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
from IPython.display import display
from numpy.random import multivariate_normal as mvn

from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
# Data split & load
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, make_moons, make_circles

def load_data(filename):
    dataset, labels = [], []
    with open(filename, 'r') as f:
        for line in f:
            x, y, label = [float(i) for i in line.strip().split()]
            dataset.append([x, y])
            labels.append(label)
    return dataset, labels

def draw_boundary(datasets, classifiers, names, is_trained=True):
    h = .02  # step size in the mesh
    figure = plt.figure(figsize=((len(classifiers))*3, len(datasets)*3))
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(classifiers)+1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            if not is_trained:
                clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            concat = np.c_[xx.ravel(), yy.ravel()]
            print(concat.shape)
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(concat)
            elif hasattr(clf, "predict_proba"):
                Z = clf.predict_proba(concat)[:, 1]
            else:
                Z = clf.predict(concat)

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                    edgecolors='k')
            # Plot the testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                    edgecolors='k', alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1

    plt.tight_layout()
    plt.show()

#%%
class LOG(Enum):
    Nope = 0
    Info = 1
    Warning = 2
    Error = 3
class SVM():
    def __init__(self, C=5, epsilon=1e-6, max_iter=1000, info_level=LOG.Info.value, kernel_type="linear", gamma=0.6):
        self.dim = 3
        self.n = 5
        self.X = np.zeros((self.n, self.dim))
        self.y = np.zeros(self.n)

        # Soft-margin 
        self.C = C

        # SMO
        self.alpha = np.zeros(self.n)
        self.b = 0
        self.epsilon = epsilon
        self.max_iter = max_iter

        # Kernel
        self.kernel_type=kernel_type
        self.gamma=gamma
        self.K = np.zeros((self.n, self.n))

        # Some utility 
        self.loss_history = []
        self.move_history = []
        self.acc_history = []
        self.inference_batch = 200
        self.use_jax = 'jax.numpy' in sys.modules
        self.info_level = info_level

    def __choose_j(self, i):
        l = list(range(self.n))
        seq = l[:i] + l[i+1:]
        return random.choice(seq)

    def __f(self, i):
        # print("Alpha: ", self.alpha.shape)
        # print("Y: ",self.y.shape)
        # print("K[i, :]: ",self.K[i, :].shape)
        # print((self.alpha * self.y * self.K[i, :]).shape)

        # return sum(self.alpha * self.y * self.K[i, :]) + self.b
        return snp.dot((self.alpha * self.y), self.K[i, :]) + self.b
    
    def __E(self, i):
        return self.__f(i) - self.y[i]

    def __eta(self, i, j):
        return self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]

    def __alpha_j_new(self, i, j):
        E_i = self.__E(i)
        E_j = self.__E(j)
        eta = self.__eta(i, j)
        return self.alpha[j] + (self.y[j] * (E_i - E_j) / eta), E_i, E_j, eta

    def __bound(self, i, j):
        # print("i: ", i, " | j: ", j, " | alpha_i: ", self.alpha[i], " | alpha_j: ", self.alpha[j], " | C: ", C)
        # print("C + alpha[j] - alpha[i]", C + self.alpha[j] - self.alpha[i])
        if self.y[i] == self.y[j]:
            B_U = min(self.C, self.alpha[j] + self.alpha[i])
            B_L = max(0, self.alpha[j] + self.alpha[i] - self.C)
        else:
            B_U = min(self.C, self.C + self.alpha[j] - self.alpha[i])
            B_L = max(0, self.alpha[j] - self.alpha[i])
        return B_U, B_L

    def __update_alpha_j(self, i, j):
        B_U, B_L = self.__bound(i, j)
        alpha_j_star, E_i, E_j, eta = self.__alpha_j_new(i, j)
        return np.clip(alpha_j_star, B_L, B_U), E_i, E_j, eta

    def __update_alpha_i(self, i, j, alpha_j_star):
        return self.alpha[i] + self.y[i] * self.y[j] * (self.alpha[j] - alpha_j_star)

    def __update_b(self, i, j, alpha_i_star, alpha_j_star, E_i, E_j):
        b_star = 0
        b_i_star = -E_i - self.y[i] * self.K[i, i] * (alpha_i_star - self.alpha[i]) - self.y[j] * self.K[j, i] * (alpha_j_star - self.alpha[j]) + self.b
        b_j_star = -E_j - self.y[i] * self.K[i, j] * (alpha_i_star - self.alpha[i]) - self.y[j] * self.K[j, j] * (alpha_j_star - self.alpha[j]) + self.b

        if alpha_i_star <= self.C and alpha_i_star >= 0:
            b_star = b_i_star
        elif alpha_j_star <= self.C and alpha_j_star >= 0:
            b_star = b_j_star
        else:
            b_star = (b_i_star + b_j_star) / 2
        
        return b_star

    def __set_alpha(self, i, val):
        if self.use_jax:
            self.alpha.at[i].set(val)
        else:
            self.alpha[i] = val

    def kernel(self, x_i, x_j, name="rbf", gamma=10):
        return np.exp(-gamma * (x_i - x_j)^2)

    def cal_kernel(self, X):
        if self.kernel_type == "rbf":
            # RBF Kernel
            # Scipy, compute exact kernel
            # pairwise_dists = squareform(pdist(X, 'euclidean'))
            # K = snp.exp(-self.gamma * (pairwise_dists ** 2))

            # Kernel Approx
            n = snp.array(X).shape[0]
            d = snp.array(X).shape[1]
            sample_n = 100*d
            W = np.random.normal(loc=0, scale=1, size=(sample_n, self.dim))
            b = np.random.uniform(0, 2*np.pi, size=sample_n)
            B = np.repeat(b[:, snp.newaxis], n, axis=1)
            norm = 1./ snp.sqrt(sample_n)
            Z = norm * snp.sqrt(2) * snp.cos(snp.dot(W, X.T) + B)
            K = np.array(snp.dot(Z.T, Z))
            return K
        else:
            # Linear Kernel
            K = snp.dot(X, X.T)
            return K

    def fit(self, X, y):
        # self.info_level = info_level
        self.X = np.array(X)
        self.y = np.reshape(np.array(y), (-1, ))
        self.n, self.dim = self.X.shape
        
        # self.kernel_type = kernel_type
        # self.gamma = gamma
        self.K = self.cal_kernel(self.X)

        self.alpha = np.zeros(self.n)
        self.b = 0

        iter = 0
        loss = np.inf
        move = np.inf

        while iter < self.max_iter and move > self.epsilon:
            loss = move = 0

            for i in range(self.n):
                j = self.__choose_j(i)

                alpha_j_star, E_i, E_j, eta = self.__update_alpha_j(i, j)
                if eta <= 0:
                    self.warning('Eta <= 0')
                    continue

                alpha_i_star = self.__update_alpha_i(i, j, alpha_j_star)
                if abs(alpha_j_star - self.alpha[j]) < 0.00001:
                    self.warning('alpha_j not moving enough')
                    continue

                b_star = self.__update_b(i, j, alpha_i_star, alpha_j_star, E_i, E_j)

                # Calculate the movement of alpha and b
                move = move + abs(alpha_i_star - self.alpha[i]) + abs(alpha_j_star - self.alpha[j]) + abs(b_star - self.b)

                # Update variables
                self.alpha[i] = alpha_i_star
                self.alpha[j] = alpha_j_star
                # self.__set_alpha(i, alpha_i_star)
                # self.__set_alpha(j, alpha_j_star)
                self.b = b_star
            
            # Calculate the loss
            loss = sum(map(lambda x: abs(self.__E(x)), np.arange(self.n)))
            # Calculate the accuracy
            acc = self.acc()
            self.loss_history.append(loss)
            self.move_history.append(move)
            self.acc_history.append(acc)

            # if not skip:
            iter += 1
            self.info("Iter: ", iter, " | Loss: ", loss, " | Move: ", move, " | Acc: ", acc)

    def decision_function(self, X):
        def make_decision(X):
            X_c = np.concatenate((self.X, X))
            K_c = self.cal_kernel(X_c)
            K_train_test = K_c[0:self.n, self.n:]
            decision = np.dot((self.alpha * self.y), K_train_test) + self.b
            return list(decision)
        
        # Split the input data into several batches
        m = np.array(X).shape[0]
        X_batch = np.array_split(X, m/self.inference_batch + 1)
        decisions = list(map(make_decision, X_batch))
        decisions = np.reshape(np.hstack(decisions), (-1, ))
        return decisions

    def predict(self, X):
        pred = self.decision_function(X)
        pred[pred > 0] = 1
        pred[pred < 0] = -1

        return pred

    def acc(self):
        idxs = np.arange(self.n)
        accs = self.y[idxs] * self.__f(idxs)
        acc = sum(accs > 0) / self.n

        return acc
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_pred, y_test = np.array(y_pred).ravel(), np.array(y_test).ravel()
        n = y_test.shape[0]
        acc = sum((y_pred * y_test) > 0) / n

        return acc

    def test(self):
        print(self.__f(0))

    # Logging
    def info(self, *args):
        if self.info_level >= LOG.Info.value:
            print("INFO: ", *args)

    def warning(self, *args):
        if self.info_level >= LOG.Warning.value:
            print("WARNING: ", *args)

    def error(self, *args):
        if self.info_level >= LOG.Error.value:
            print("ERROR: ", *args)
    
    # Save pickle
    def save(self, name, protocol=pickle.HIGHEST_PROTOCOL):
        with open(name, 'wb') as handle:
            pickle.dump(self, handle, protocol=protocol)

    def load(self, name):
        with open(name, 'rb') as handle:
            b = pickle.load(handle)
            return b


def gen_dataset():
    size = 50
    mean = [2, 2]
    cov = [[1, 0], [0, 1]]
    X1 = mvn(mean, cov, size)
    y1 = np.full(size, -1)

    mean = [-2, -2]
    cov = [[1, 0], [0, 1]]
    X2 = mvn(mean, cov, size)
    y2 = np.full(size, 1)

    return np.concatenate((X1, X2)), np.concatenate((y1, y2))

def acc_rate(y_pred, y_test):
    y_pred, y_test = np.array(y_pred).ravel(), np.array(y_test).ravel()
    n = y_test.shape[0]
    acc = sum((y_pred * y_test) > 0) / n

    return acc

def acc(svm, X_train, X_test, y_train, y_test):
    train_pred = svm.predict(X_train)
    train_acc = svm.score(X_train, y_train)
    test_pred = svm.predict(X_test)
    test_acc = svm.score(X_test, y_test)

    # train_pred = svm.predict(X_train)
    # train_acc = acc_rate(train_pred, y_train)
    # test_pred = svm.predict(X_test)
    # test_acc = acc_rate(test_pred, y_test)

    print("Train Acc: ", train_acc, " | Test Acc: ", test_acc)
    return train_pred, test_pred, train_acc, test_acc

def load_bc():
    dataset, labels = load_breast_cancer(return_X_y=True)
    labels[labels == 0] = -1

    return dataset, labels

def load_moon():
    dataset, labels = make_moons(n_samples=150)
    labels[labels == 0] = -1

    return dataset, labels

def load_circle():
    dataset, labels = make_circles(noise=0.2, factor=0.5, random_state=1)
    labels[labels == 0] = -1

    return dataset, labels

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
from sklearn.decomposition import LatentDirichletAllocation, PCA

def vec(review_text):
    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, norm='l2')
    count = CountVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

    # vec_text = count.fit_transform(review_text)
    vec_text = np.array(tfidf.fit_transform(review_text).toarray())

    return vec_text, tfidf

def lda(vec_text, n_comp):
    lda = LatentDirichletAllocation(n_components = n_comp, random_state = 0)
    decom_text = lda.fit_transform(vec_text)

    pd.DataFrame(decom_text).head()

    return decom_text, lda

def pca(vec_text, n_comp):
    pca = PCA(n_components=n_comp)
    pca.fit(vec_text)
    decom_text = pca.transform(vec_text)

    pd.DataFrame(decom_text).head()

    return decom_text, pca
# %%
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
def plot_top_words(model, vectorizer, n_top_words, title):
    feature_names = vectorizer.get_feature_names()
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
is_read_original = True
if is_read_original:
    df = prep_save()
else:
    df = pd.read_pickle("clean.pkl")
    df_original = pd.read_pickle("no_null.pkl")
# %%
# Condition slicing
# Small datset for debugging
# df = df.sample(frac=0.1, replace=False, random_state=42)

# df = df[df["Recommended IND"] == 1]
# df = df[df["Rating"] == 1]

# df = df[df["Department Name"] == "Intimate"]

# df = df[df["Class Name"] == "Blouses"]
# df = df[df["Class Name"] == "Bottoms"]

# df = df[df["Age"] >= 60]
# df = df[df["Age"] < 60]

df.head()
# %%
# Test SVM
datasets = [gen_dataset(), load_moon(), load_circle()]
svm = SVM(C=0.6, max_iter=1000, kernel_type="rbf", gamma=2)
draw_boundary(datasets, [SVM(C=0.6, max_iter=1000, kernel_type="rbf", gamma=2), SVM(C=0.6, max_iter=1000, kernel_type="linear")], ["RBF SVM", "Linear SVM"], is_trained=False)

# %%
# Vectorization
vec_text, vectorizer = vec(df['Review Text'])

# Dimension Reduction
decom_text, lda = lda(vec_text, n_comp=5)
# decom_text, pca = pca(vec_text, n_comp=5)

# %%
# Plot distribution of properties
plot_dist(df)

# Average age & rating of each group
groups = ['Division Name', 'Department Name', 'Class Name']
fields = ['Age', 'Rating']

for group in groups:
    for field in fields:
        print("Group By: ", group)
        display(pd.DataFrame(df[[group, field]]).groupby(group).mean())

# %%
# Vectorizer features
summary_text = np.sum(vec_text, axis=0)
vectorizer._validate_vocabulary()
display(pd.DataFrame(summary_text, index=vectorizer.get_feature_names()).nlargest(10, columns=[0]))

# Top words of LDA topics
n_top_words = 10
plot_top_words(lda, vectorizer, n_top_words, 'Topics in LDA model')

# Word Cloud
cloud = WordCloud().generate(" ".join(list(df['Review Text'])))
cloud.to_file('output.png')

# %%
# Select review text of specific topic
# df_lda = pd.DataFrame(decom_text)
# display(df_lda)
# df_txt = pd.DataFrame(df_original['Review Text'].iloc[np.argmax(decom_text, axis=1)== 1])
# df_txt.to_csv("topic2_reviews.csv")
# display(df_txt)
# %%
from sklearn.model_selection import train_test_split
# from svm import SVM as SVM, acc, draw_boundary

def label_proc(labels):
    labels[labels == 0] = -1
    return labels

svm = SVM(C=0.6, max_iter=1000, kernel_type="rbf", gamma=2, epsilon=50)
X, y = decom_text, label_proc(df["Recommended IND"].to_numpy())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

svm.fit(X_train, y_train)

train_pred, test_pred, train_acc, test_acc = acc(svm, X_train, X_test, y_train, y_test)

display(test_pred[:10])

# draw_boundary([(X, y)], [svm], ["RBF SVM"])

# %%
# svm.save("svm_pca5.pickle")
svm.save("svm_lda5.pickle")

# svm2 = svm.load("svm_pca5.pickle")

def plot_training(svm, text=''):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4))
    fig.suptitle('SVM During Training'+text)

    ax1.plot(svm.acc_history)
    ax1.title.set_text("Training Accuracy")
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True)

    ax2.plot(svm.loss_history)
    ax2.title.set_text("Loss")
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.grid(True)

    ax3.plot(svm.move_history)
    ax3.title.set_text("Movement of Variables")
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('change of variables')
    ax3.grid(True)

    fig.tight_layout()

plot_training(svm, text=' With LDA Preprocess')
# plot_training(svm2, text=' With PCA Preprocess')
# %%
