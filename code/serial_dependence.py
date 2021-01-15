import numpy as np
import mne
from sklearn.model_selection import KFold

import load_data as ld
import machine_learning as ml
import matplotlib.pyplot as plt

def get_diffs(subj, n=500):
    _, labels = ld.load_behavior(subj)
    labels = labels[0, :n]
    diffs = [0]
    for i in range(1, 500):
        ors_diff = np.abs(labels[i-1] - labels[i])
        diffs.append(ors_diff)

    diffs = np.array(diffs)
    return diffs

def analyze_selectivity(subj, tmin=0, tmax=16):
    diffs = get_diffs(subj)

    bins = {}
    for i in range(18):
        bins[i] = []

    X, _, y, _ = ld.load_data(subj, data="epochs", n_train=500, n_test=0)

    kfold = KFold(n_splits=5)
    for train, test in kfold.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        #model = ml.LogisticSlidingModel(max_iter=4000, n_classes=4, k=1000, C=0.05, l1_ratio=0.95)
        model = ml.LogisticSlidingModel(max_iter=4000, n_classes=4, k=20, C=0.1, l1_ratio=0.95)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        split_diffs = diffs[test]
        for i in range(100):
            acc = 0
            for j in range(tmin, tmax):
                if pred[i][j] == y_test[i]:
                    acc += 1
            acc /= (tmax - tmin)

            bin_idx = split_diffs[i] // 10
            bins[bin_idx].append(acc)

    bin_accuracies = []
    for i in range(18):
        acc = np.mean(np.array(bins[i]))
        bin_accuracies.append(acc)

    plt.bar(np.arange(18), bin_accuracies)
    plt.savefig("../Figures/SD/selectivity/sd_accuracy_%s_%d_%d.png" % (subj, tmin, tmax))
    plt.clf()
    return bins


def analyze_bias(subj, tmin, tmax):
    # x-axis: difference between current and previous orientation (bins?)
    # y-axis: decoding bias, i.e. difference between truth and predicted value
    
    diffs = get_diffs(subj)

    X, _, y, _ = ld.load_data(subj, data="epochs", n_train=500, n_test=0)

    bins = []

    kfold = KFold(n_splits=5)
    for train, test in kfold.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        #model = ml.LogisticSlidingModel(max_iter=4000, n_classes=4, k=1000, C=0.05, l1_ratio=0.95)
        model = ml.LogisticSlidingModel(max_iter=4000, n_classes=4, k=20, C=0.1, l1_ratio=0.95)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        split_diffs = diffs[test]
        for i in range(100):
            for j in range(tmin, tmax):
                bias = pred[i][j] - y_test[i]
                if bias == 3:
                    bias = -1
                elif bias == -3:
                    bias = 1
                bins.append([split_diffs[i], bias])

    bins = np.array(bins)
    plt.scatter(bins[:, 0], bins[:, 1])
    plt.savefig("../Figures/SD/bias/sd_accuracy_%s_%d_%d.png" % (subj, tmin, tmax))
    return bins

def split_half_analysis(subj):
    X, X_test, y, y_test = ld.load_data(subj, data="epochs", n_train=400, n_test=200)
    X_close, y_close = [], []
    X_far, y_far = [], []

    diffs = get_diffs(subj, n=600)
    diffs = diffs[-200:]

    for i in range(200):
        if diffs[i] < 45 or diffs[i] >= 135 :
            X_close.append(X_test[i])
            y_close.append(y_test[i])
        else:
            X_far.append(X_test[i])
            y_far.append(y_test[i])

    X_close, y_close = np.array(X_close), np.array(y_close)
    X_far, y_far = np.array(X_far), np.array(y_far)

    model = ml.LogisticSlidingModel(max_iter=4000, n_classes=4, k=20, C=0.1, l1_ratio=0.95)
    model.fit(X, y)
    close_pred = model.evaluate(X_close, y_close)
    far_pred = model.evaluate(X_far, y_far)

    plt.plot(np.arange(0, 0.4, 0.025), close_pred, label="close")
    plt.plot(np.arange(0, 0.4, 0.025), far_pred, label='far')
    plt.ylim((0.2, 0.4))
    plt.legend()
    plt.savefig('../Figures/SD/split_half/split_half_%s.png' % subj)
    plt.clf()

    return close_pred, far_pred










