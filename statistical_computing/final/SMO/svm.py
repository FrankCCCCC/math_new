#%%
import random
import numpy as np
from scipy.spatial.distance import pdist, squareform

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from numpy.random import multivariate_normal as mvn

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
#%%
def load_data(filename):
    dataset, labels = [], []
    with open(filename, 'r') as f:
        for line in f:
            x, y, label = [float(i) for i in line.strip().split()]
            dataset.append([x, y])
            labels.append(label)
    return dataset, labels

def get_w(alphas, dataset, labels):
    ''' 通过已知数据点和拉格朗日乘子获得分割超平面参数w
    '''
    alphas, dataset, labels = np.array(alphas), np.array(dataset), np.array(labels)
    yx = labels.reshape(1, -1).T*np.array([1, 1])*dataset
    w = np.dot(yx.T, alphas)

    return w.tolist()

def draw_boundary(dataset, labels, alphas, b):
    # 分类数据点
    classified_pts = {'+1': [], '-1': []}
    for point, label in zip(dataset, labels):
        if label == 1.0:
            classified_pts['+1'].append(point)
        else:
            classified_pts['-1'].append(point)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 绘制数据点
    for label, pts in classified_pts.items():
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], label=label)

    # 绘制分割线
    w = get_w(alphas, dataset, labels)
    x1, _ = max(dataset, key=lambda x: x[0])
    x2, _ = min(dataset, key=lambda x: x[0])
    a1, a2 = w
    y1, y2 = (-b - a1*x1)/a2, (-b - a1*x2)/a2
    ax.plot([x1, x2], [y1, y2])

    # 绘制支持向量
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 1e-3:
            x, y = dataset[i]
            ax.scatter([x], [y], s=150, c='none', alpha=0.7,
                       linewidth=1.5, edgecolor='#AB3319')

    plt.show()

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
        self.debug_mode = False
        self.acc_history = []

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
        return np.dot((self.alpha * self.y), self.K[i, :]) + self.b
    
    def __E(self, i):
        return self.__f(i) - self.y[i]

    def __eta(self, i, j):
        return self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]

    def __alpha_j_new(self, i, j):
        E_i = self.__E(i)
        E_j = self.__E(j)
        eta = self.__eta(i, j)
        return self.alpha[j] + (self.y[j] * (E_i - E_j) / eta), E_i, E_j, eta

    def __bound(self, i, j, C):
        # print("i: ", i, " | j: ", j, " | alpha_i: ", self.alpha[i], " | alpha_j: ", self.alpha[j], " | C: ", C)
        # print("C + alpha[j] - alpha[i]", C + self.alpha[j] - self.alpha[i])
        if self.y[i] == self.y[j]:
            B_U = min(C, self.alpha[j] + self.alpha[i])
            B_L = max(0, self.alpha[j] + self.alpha[i] - C)
        else:
            B_U = min(C, C + self.alpha[j] - self.alpha[i])
            B_L = max(0, self.alpha[j] - self.alpha[i])
        return B_U, B_L

    def __update_alpha_j(self, i, j, C):
        B_U, B_L = self.__bound(i, j, C)
        alpha_j_star, E_i, E_j, eta = self.__alpha_j_new(i, j)
        return np.clip(alpha_j_star, B_L, B_U), E_i, E_j, eta

    def __update_alpha_i(self, i, j, alpha_j_star):
        return self.alpha[i] + self.y[i] * self.y[j] * (self.alpha[j] - alpha_j_star)

    def __update_b(self, i, j, alpha_i_star, alpha_j_star, E_i, E_j, C):
        b_star = 0
        b_i_star = -E_i - self.y[i] * self.K[i, i] * (alpha_i_star - self.alpha[i]) - self.y[j] * self.K[j, i] * (alpha_j_star - self.alpha[j]) + self.b
        b_j_star = -E_j - self.y[i] * self.K[i, j] * (alpha_i_star - self.alpha[i]) - self.y[j] * self.K[j, j] * (alpha_j_star - self.alpha[j]) + self.b

        if alpha_i_star <= C and alpha_i_star >= 0:
            b_star = b_i_star
        elif alpha_j_star <= C and alpha_j_star >= 0:
            b_star = b_j_star
        else:
            b_star = (b_i_star + b_j_star) / 2
        
        return b_star

    def kernel(self, x_i, x_j, name="rbf", gamma=10):
        return np.exp(-gamma * (x_i - x_j)^2)

    def cal_kernel(self, X, gamma=10):
        # RBF Kernel
        # pairwise_dists = squareform(pdist(X, 'euclidean'))
        # K = np.exp(-gamma * (pairwise_dists ** 2))

        # Linear Kernel
        K = np.dot(X, X.T)
        return K

    def fit(self, X, y, C=5, epsilon=1e-6, max_iter=1000, debug_mode=False):
        self.debug_mode = debug_mode
        self.n, self.dim = np.array(X).shape
        self.X = np.array(X)
        self.y = np.reshape(np.array(y), (-1, ))
        self.K = self.cal_kernel(X)

        self.alpha = np.zeros(self.n)
        self.b = 0

        iter = 0
        loss = np.inf
        move = np.inf

        while iter < max_iter and move > epsilon:
            loss = move = 0
            # skip = False

            for i in range(self.n):
                j = self.__choose_j(i)

                alpha_j_star, E_i, E_j, eta = self.__update_alpha_j(i, j, C)
                if eta <= 0:
                    self.log('WARNING: Eta <= 0')
                    # skip = True
                    continue

                alpha_i_star = self.__update_alpha_i(i, j, alpha_j_star)
                if abs(alpha_j_star - self.alpha[j]) < 0.00001:
                    self.log('WARNING: alpha_j not moving enough')
                    # skip = True
                    continue

                b_star = self.__update_b(i, j, alpha_i_star, alpha_j_star, E_i, E_j, C)

                # Calculate the movement of alpha and b
                move = move + abs(alpha_i_star - self.alpha[i]) + abs(alpha_j_star - self.alpha[j]) + abs(b_star - self.b)

                # Update variables
                self.alpha[i] = alpha_i_star
                self.alpha[j] = alpha_j_star
                self.b = b_star
            
            # Calculate the loss
            loss = sum(map(lambda x: abs(self.__E(x)), np.arange(self.n)))
            # Calculate the accuracy
            acc = self.acc()
            self.acc_history.append(acc)

            # if not skip:
            iter += 1
            print("Iter: ", iter, " | Loss: ", loss, " | Move: ", move, " | Acc: ", acc)

    def predict(self, X):
        X_c = np.concatenate((self.X, X))
        K_c = self.cal_kernel(X_c)
        K_train_test = K_c[0:self.n, self.n:]
        pred = np.dot((self.alpha * self.y), K_train_test) + self.b
        return pred

    def acc(self):
        idxs = np.arange(self.n)
        accs = self.y[idxs] * self.__f(idxs)
        acc = sum(accs > 0) / self.n

        return acc

    def test(self):
        print(self.__f(0))

    def log(self, *args, **kwargs):
        if self.debug_mode:
            print(args, kwargs)


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

def acc(X_train, X_test, y_train, y_test):
    train_pred = svm.predict(X_train)
    train_acc = acc_rate(train_pred, y_train)
    test_pred = svm.predict(X_test)
    test_acc = acc_rate(test_pred, y_test)

    print("Train Acc: ", train_acc, " | Test Acc: ", test_acc)
    return train_pred, test_pred, train_acc, test_acc

def load_bc():
    dataset, labels = load_breast_cancer(return_X_y=True)
    labels[labels == 0] = -1

    return dataset, labels
#%%
if __name__ == "__main__":
    # dataset, labels = load_data('testSet.txt')
    # dataset, labels = load_bc()
    dataset, labels = gen_dataset()

    X, y = np.array(dataset), np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    print("X_Train: ", X_train.shape)
    display(pd.DataFrame(X_train).head())
    print("Y_Train: ", y_train.shape)
    display(pd.DataFrame(y_train).head())

    svm = SVM()

    svm.fit(X_train, y_train, C=0.6, max_iter=1000)

    train_pred, test_pred, train_acc, test_acc = acc(X_train, X_test, y_train, y_test)

    display(test_pred[:10])

    # draw_boundary(X_train, y_train, svm.alpha, svm.b)
