#%%
try:
    import jax.numpy as snp
    print("Use Jax.numpy module")
except:
    import numpy as snp
    print("Didn't find out Jax module, use numpy instead")

import numpy as np

import pickle
import sys
from enum import Enum
import random
from scipy.spatial.distance import pdist, squareform

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from numpy.random import multivariate_normal as mvn

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, make_moons, make_circles
#%%
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler

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
    figure = plt.figure(figsize=((len(classifiers))*3, len(datasets)*2))
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
        # kernel_type = "linear" or "rbf"
        self.kernel_type=kernel_type
        self.gamma=gamma
        self.K = np.zeros((self.n, self.n))
        # Set up kernel function
        self.kernel_method = getattr(self, self.kernel_type)

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

    def rbf(self, X):
        # RBF Kernel
        # Scipy, compute exact kernel
        pairwise_dists = squareform(pdist(X, 'euclidean'))
        K = snp.exp(-self.gamma * (pairwise_dists ** 2))
        return K
    
    def rbf_approx(self, X):
        # Kernel Approx
        n = snp.array(X).shape[0]
        d = snp.array(X).shape[1]
        sample_n = 100*d
        W = np.random.normal(loc=0, scale=self.gamma/2, size=(sample_n, self.dim))
        b = np.random.uniform(0, 2*np.pi, size=sample_n)
        B = np.repeat(b[:, snp.newaxis], n, axis=1)
        norm = 1./ snp.sqrt(sample_n)
        Z = norm * snp.sqrt(2) * snp.cos(snp.dot(W, X.T) + B)
        K = np.array(snp.dot(Z.T, Z))
        return K
    
    def linear(self, X):
        # Linear Kernel
        K = snp.dot(X, X.T)
        return K

    def cal_kernel(self, X):
        return self.kernel_method(X)

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
#%%
if __name__ == "__main__":
    # dataset, labels = load_data('testSet.txt')
    # dataset, labels = load_bc()
    # dataset, labels = gen_dataset()
    # dataset, labels = load_moon()
    # dataset, labels = load_circle()

    datasets = [load_data('testSet.txt'), load_moon(), load_circle()]

    # X, y = np.array(dataset), np.array(labels)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    # print("X_Train: ", X_train.shape)
    # display(pd.DataFrame(X_train).head())
    # print("Y_Train: ", y_train.shape)
    # display(pd.DataFrame(y_train).head())

    svm = SVM(C=0.6, max_iter=1000, kernel_type="rbf", gamma=2)

    # svm.fit(X_train, y_train, C=0.6, max_iter=1000, kernel_type="rbf", gamma=2)

    # train_pred, test_pred, train_acc, test_acc = acc(svm, X_train, X_test, y_train, y_test)

    # display(test_pred[:10])

    # draw_boundary([(dataset, labels)], [svm], ["RBF SVM"], is_trained=True)
    # draw_boundary(datasets, [SVM(C=0.6, max_iter=1000, kernel_type="rbf", gamma=2), SVM(C=0.6, max_iter=1000, kernel_type="linear")], ["RBF SVM", "Linear SVM"], is_trained=False)
    draw_boundary(datasets, [SVM(C=0.6, max_iter=1000, kernel_type="rbf_approx", gamma=2), SVM(C=0.6, max_iter=1000, kernel_type="linear")], ["RBF SVM", "Linear SVM"], is_trained=False)
