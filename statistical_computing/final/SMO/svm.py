#%%
import numpy as np
from scipy.spatial.distance import pdist, squareform

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

#%%
def load_data(filename):
    dataset, labels = [], []
    with open(filename, 'r') as f:
        for line in f:
            x, y, label = [float(i) for i in line.strip().split()]
            dataset.append([x, y])
            labels.append(label)
    return dataset, labels

#%%
class SVM():
    def __init__(self):
        self.dim = 3
        self.n = 5
        self.alpha = np.zeros(self.n)
        self.b = 0
        self.X = np.zeros((self.n, self.dim))
        self.y = np.zeros(self.n)
        self.K = np.zeros((self.n, self.n))
        self.embed = np.zeros((self.n, self.dim))

    def __f(self, i):
        return sum(self.alpha * self.y * self.K[i, :]) + self.b
    
    def __E(self, i):
        return self.__f(i) - self.y[i]

    def __eta(self, i, j):
        return self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]

    def __alpha_j_new(self, i, j):
        return self.alpha[j] + (self.y[j] * (self.__E(i) - self.__E(j) / self.__eta(i, j)))

    def __bound(self, i, j, C):
        if self.y[i] == self.y[j]:
            B_U = np.min(C, self.alpha[j] + self.alpha[i])
            B_L = np.max(0, self.alpha[j] - self.alpha[i] - C)
            return B_U, B_L
        else:
            B_U = np.min(C, C + self.alpha[j] - self.alpha[i])
            B_L = np.max(0, self.alpha[j] - self.alpha[i])
            return B_U, B_L

    def kernel(self, x_i, x_j, name="rbf", gamma=10):
        return np.exp(-gamma * (x_i - x_j)^2)

    def cal_kernel(self, X, gamma=10):
        pairwise_dists = squareform(pdist(X, 'euclidean'))
        K = np.exp(-gamma * (pairwise_dists ** 2))
        return K

    def fit(self, X, y):
        self.n, self.dim = np.array(X).shape
        self.X = np.array(X)
        self.y = np.array(y)
        self.K = self.cal_kernel(X)

    def test(self):
        print(self.__f(0))

#%%
if __name__ == "__main__":
    dataset, labels = load_data('testSet.txt')
    X_df = pd.DataFrame(dataset)
    y_df = pd.DataFrame(labels)

    display(X_df.head())
    display(y_df.head())

    svm = SVM()
    svm.test()
    # display(svm.cal_kernel(X_df.to_numpy()[:5, :]))
    


# %%
