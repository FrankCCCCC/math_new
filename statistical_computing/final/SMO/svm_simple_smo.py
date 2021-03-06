#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    dataset, labels = [], []
    with open(filename, 'r') as f:
        for line in f:
            x, y, label = [float(i) for i in line.strip().split()]
            dataset.append([x, y])
            labels.append(label)
    return dataset, labels

def clip(alpha, L, H):
    ''' 修建alpha的值到L和H之间.
    '''
    if alpha < L:
        return L
    elif alpha > H:
        return H
    else:
        return alpha

def select_j(i, m):
    ''' 在m中随机选择除了i之外剩余的数
    '''
    l = list(range(m))
    seq = l[: i] + l[i+1:]
    return random.choice(seq)

def get_w(alphas, dataset, labels):
    ''' 通过已知数据点和拉格朗日乘子获得分割超平面参数w
    '''
    alphas, dataset, labels = np.array(alphas), np.array(dataset), np.array(labels)
    yx = labels.reshape(1, -1).T*np.array([1, 1])*dataset
    w = np.dot(yx.T, alphas)

    return w.tolist()

def simple_smo(dataset, labels, C, max_iter):
    ''' 简化版SMO算法实现，未使用启发式方法对alpha对进行选择.

    :param dataset: 所有特征数据向量
    :param labels: 所有的数据标签
    :param C: 软间隔常数, 0 <= alpha_i <= C
    :param max_iter: 外层循环最大迭代次数
    '''
    dataset = np.array(dataset)
    m, n = dataset.shape
    labels = np.array(labels)

    # 初始化参数
    alphas = np.zeros(m)
    b = 0
    it = 0
    K = np.dot(dataset, dataset.T)

    # def f(x):
    #     "SVM分类器函数 y = w^Tx + b"
    #     # Kernel function vector.
    #     x = np.matrix(x).T
    #     data = np.matrix(dataset)
    #     ks = data*x

    #     # Predictive value.
    #     wx = np.matrix(alphas*labels)*ks
    #     fx = wx + b

    #     return fx[0, 0]
    
    def f(i):
        "SVM分类器函数 y = w^Tx + b"
        # Kernel function vector.
        ks = K[i,]

        # Predictive value.
        wx = np.matrix(alphas*labels)*ks
        fx = wx + b

        return fx[0, 0]

    def __f(i):
        return sum(alphas * labels * K[i, :]) + b

    def __E(i):
        return __f(i) - labels[i]

    def __eta(i, j):
        return K[i, i] + K[j, j] - 2 * K[i, j]
    
    def __alpha_j_new(i, j):
        E_i = __E(i)
        E_j = __E(j)
        eta = __eta(i, j)
        return alphas[j] + (labels[j] * (E_i - E_j) / eta), E_i, E_j, eta

    def __bound(i, j, C):
        if labels[i] == labels[j]:
            B_U = min(C, alphas[j] + alphas[i])
            B_L = max(0, alphas[j] + alphas[i] - C)
        else:
            B_U = min(C, C + alphas[j] - alphas[i])
            B_L = max(0, alphas[j] - alphas[i])
        return B_U, B_L

    def __update_alpha_j(i, j, C):
        B_U, B_L = __bound(i, j, C)
        alpha_j_star, E_i, E_j, eta = __alpha_j_new(i, j)
        return np.clip(alpha_j_star, B_L, B_U), E_i, E_j, eta

    def __update_alpha_i(i, j, alpha_j_star):
        return alphas[i] + labels[i] * labels[j] * (alphas[j] - alpha_j_star)

    def __update_b(i, j, alpha_i_star, alpha_j_star, E_i, E_j, C):
        b_star = 0
        b_i_star = -E_i - labels[i] * K[i, i] * (alpha_i_star - alphas[i]) - labels[j] * K[j, i] * (alpha_j_star - alphas[j]) + b
        b_j_star = -E_j - labels[i] * K[i, j] * (alpha_i_star - alphas[i]) - labels[j] * K[j, j] * (alpha_j_star - alphas[j]) + b

        if alpha_i_star <= C and alpha_i_star >= 0:
            b_star = b_i_star
        elif alpha_j_star <= C and alpha_j_star >= 0:
            b_star = b_j_star
        else:
            b_star = (b_i_star + b_j_star) / 2
        
        return b_star

    all_alphas, all_bs = [], []

    while it < max_iter:
        pair_changed = 0
        for i in range(m):
            # a_i, x_i, y_i = alphas[i], dataset[i], labels[i]
            # fx_i = f(x_i)
            # fx_i = __f(i)
            # E_i = fx_i - y_i

            j = select_j(i, m)
            # a_j, x_j, y_j = alphas[j], dataset[j], labels[j]
            # fx_j = f(x_j)
            # fx_j = __f(j)
            # E_j = fx_j - y_j

            # K_ii, K_jj, K_ij = np.dot(x_i, x_i), np.dot(x_j, x_j), np.dot(x_i, x_j)
            # K_ii, K_jj, K_ij = K[i, i], K[j, j], K[i, j]
            # eta = K_ii + K_jj - 2*K_ij
            # eta = __eta(i, j)
            # if eta <= 0:
            #     print('WARNING  eta <= 0')
            #     continue

            # 获取更新的alpha对
            # a_i_old, a_j_old = a_i, a_j
            # a_j_new = a_j_old + y_j*(E_i - E_j)/eta

            # 对alpha进行修剪
            # if y_i != y_j:
            #     L = max(0, a_j_old - a_i_old)
            #     H = min(C, C + a_j_old - a_i_old)
            # else:
            #     L = max(0, a_i_old + a_j_old - C)
            #     H = min(C, a_j_old + a_i_old)

            # a_j_new = clip(a_j_new, L, H)
            a_j_new, E_i, E_j, eta = __update_alpha_j(i, j, C)
            if eta <= 0:
                print('WARNING  eta <= 0')
                continue
            # a_i_new = a_i_old + y_i*y_j*(a_j_old - a_j_new)
            a_i_new = __update_alpha_i(i, j, a_j_new)

            if abs(a_j_new - alphas[j]) < 0.00001:
                print('WARNING   alpha_j not moving enough')
                continue

            # 更新阈值b
            #import ipdb; ipdb.set_trace()
            # b_i = -E_i - y_i*K_ii*(a_i_new - a_i_old) - y_j*K_ij*(a_j_new - a_j_old) + b
            # b_j = -E_j - y_i*K_ij*(a_i_new - a_i_old) - y_j*K_jj*(a_j_new - a_j_old) + b

            # if 0 < a_i_new < C:
            #     b = b_i
            # elif 0 < a_j_new < C:
            #     b = b_j
            # else:
            #     b = (b_i + b_j)/2

            b = __update_b(i, j, a_i_new, a_j_new, E_i, E_j, C)

            alphas[i], alphas[j] = a_i_new, a_j_new

            all_alphas.append(alphas)
            all_bs.append(b)

            pair_changed += 1
            print('INFO   iteration:{}  i:{}  pair_changed:{}'.format(it, i, pair_changed))

        if pair_changed == 0:
            it += 1
        else:
            it = 0
        print('iteration number: {}'.format(it))

    return alphas, b

if '__main__' == __name__:
    # 加载训练数据
    dataset, labels = load_data('testSet.txt')
    # 使用简化版SMO算法优化SVM
    alphas, b = simple_smo(dataset, labels, 0.6, 40)

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

