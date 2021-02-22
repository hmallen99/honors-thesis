import numpy as np
import mne
from sklearn.model_selection import KFold

import load_data as ld
import machine_learning as ml
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D



def calc_relative_orientation(x):
    if np.abs(x) > 90:
        x = x - (np.sign(x) * 180)
        return x
    return x

def get_diffs(subj, n=500, shift=-1):
    if shift == 0:
        return np.zeros(n)
    _, labels = ld.load_behavior(subj)
    labels = labels[0, :n]
    if shift < 0:
        shift = shift * -1
        diffs = [0 for i in range(shift)]
        for i in range(shift, n):
            ors_diff = calc_relative_orientation(labels[i-shift] - labels[i])
            diffs.append(ors_diff + 90)
        return np.array(diffs)
    else:
        diffs = []
        for i in range(0, n-shift):
            ors_diff = calc_relative_orientation(labels[i + shift] - labels[i])
            diffs.append(ors_diff + 90)
        diffs.extend([0 for i in range(shift)])
        return np.array(diffs)

def analyze_serial_dependence(subj, n=500):
    res, tgt = ld.load_behavior(subj)
    res, tgt = res[0, :n], tgt[0, :n]
    rel_ors = [0]
    first_err = calc_relative_orientation(res[0] - tgt[0])
    errors = [first_err]
    for i in range(1, n):
        next_or = calc_relative_orientation(tgt[i-1] - tgt[i])
        next_er = calc_relative_orientation(res[i] - tgt[i])
        if np.abs(next_or) < 60 and np.abs(next_er) < 40:
            rel_ors += [next_or]
            errors += [next_er]


    plt.scatter(rel_ors, errors)
    plt.xlim((-60, 60))
    plt.ylim((-40, 40))
    plt.savefig("../Figures/SD/subj_sd/%s_sd.png" % subj)
    plt.clf()
    return rel_ors, errors

def analyze_selectivity(subj, tmin=0, tmax=16, n_bins=18, time_shift=-1, plot=False):
    diffs = get_diffs(subj, shift=time_shift)

    bins = {}
    for i in range(n_bins):
        bins[i] = []
    
    bin_width = 180 // n_bins

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
            bin_idx = np.minimum(split_diffs[i] // bin_width, n_bins - 1)
            bins[bin_idx].append(acc)

    bin_accuracies = []
    for i in range(n_bins):
        acc = np.mean(np.array(bins[i]))
        bin_accuracies.append(acc)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(n_bins), bin_accuracies)
        plt.xticks(ticks=np.arange(n_bins), labels=np.linspace(-90 + (bin_width/2), 90 - (bin_width/2), n_bins))
        plt.savefig("../Figures/SD/selectivity/sd_accuracy_%s_%d_%d.png" % (subj, tmin, tmax))
        plt.clf()
    return bins

def bias_helper(split_diffs, bin_width, pred, y_test, n_classes, n_bins, tmin, tmax, bins):
    for i in range(len(split_diffs)):
            bin_idx = (split_diffs[i]) // bin_width
            if bin_idx >= n_bins:
                bin_idx = n_bins - 1
            for j in range(tmin, tmax):
                bias = pred[i][j] - y_test[i]
                bias = ((bias + (n_classes / 2)) % n_classes) - (n_classes / 2)
                if bias == -int(n_classes / 2):
                    bias = n_classes / 2
                    #continue
                bins[bin_idx] += [bias]
    return bins

def analyze_bias(subj, tmin, tmax, n_bins, n_classes=4, time_shift=-1, plot=False):
    # x-axis: difference between current and previous orientation (bins?)
    # y-axis: decoding bias, i.e. difference between truth and predicted value
    
    diffs = get_diffs(subj, shift=time_shift)

    X, _, y, _ = ld.load_data(subj, data="epochs", n_classes=n_classes, n_train=500, n_test=0)

    bins = [[] for i in range(n_bins)]
    bin_width = 180 // n_bins

    kfold = KFold(n_splits=5)
    for train, test in kfold.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        model = ml.LogisticSlidingModel(max_iter=4000, n_classes=n_classes, k=20, C=0.085, l1_ratio=0.95)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        split_diffs = diffs[test]

        bins = bias_helper(split_diffs, bin_width, pred, y_test, n_classes, n_bins, tmin, tmax, bins)

    if plot:
        bin_accuracies = np.array([np.mean(np.array(bins[i])) for i in range(n_bins)])

        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(n_bins), bin_accuracies)
        plt.xticks(ticks=np.arange(n_bins), labels=np.linspace(-90 + (bin_width/2), 90 - (bin_width/2), n_bins))
        plt.savefig("../Figures/SD/bias/sd_accuracy_%s_%d_%d.png" % (subj, tmin, tmax))
        plt.clf()
    return bins

def analyze_bias_card_obl(subj, tmin, tmax, n_bins, time_shift=-1, plot=False):
    # x-axis: difference between current and previous orientation (bins?)
    # y-axis: decoding bias, i.e. difference between truth and predicted value
    
    diffs = get_diffs(subj, shift=time_shift)
    n_classes = 4
    X, _, y, _ = ld.load_data(subj, data="epochs", n_classes=n_classes, n_train=500, n_test=0)

    card_bins = [[] for i in range(n_bins)]
    obl_bins = [[] for i in range(n_bins)]
    bin_width = 180 // n_bins

    kfold = KFold(n_splits=5)
    for train, test in kfold.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        model = ml.LogisticSlidingModel(max_iter=4000, n_classes=n_classes, k=20, C=0.085, l1_ratio=0.95)
        model.fit(X_train, y_train)

        X_card, y_card = X_test[y_test % 2 == 0], y_test[y_test % 2 == 0]
        X_obl, y_obl = X_test[y_test % 2 == 1], y_test[y_test % 2 == 1]
        card_pred = model.predict(X_card)
        obl_pred = model.predict(X_obl)

        split_diffs = diffs[test]
        card_diffs = split_diffs[y_test % 2 == 0]
        obl_diffs = split_diffs[y_test % 2 == 1]
        
        card_bins = bias_helper(card_diffs, bin_width, card_pred, y_card, n_classes, n_bins, tmin, tmax, card_bins)
        obl_bins = bias_helper(obl_diffs, bin_width, obl_pred, y_obl, n_classes, n_bins, tmin, tmax, obl_bins)
          

    if plot:
        card_bin_accuracies = np.array([np.mean(np.array(card_bins[i])) for i in range(n_bins)])
        obl_bin_accuracies = np.array([np.mean(np.array(obl_bins[i])) for i in range(n_bins)])

        fig, axes = plt.subplots(nrows=2, ncols=1)
        fig.set_figheight(15)
        fig.set_figwidth(10)
        plt.setp(axes, xticks=np.arange(n_bins), xticklabels=np.linspace(-90 + (bin_width/2), 90 - (bin_width/2), n_bins))
        ax1 = axes[0]
        ax1.bar(np.arange(n_bins), card_bin_accuracies)
        ax1.title.set_text("Cardinal")

        ax2 = axes[1]
        ax2.bar(np.arange(n_bins), obl_bin_accuracies)
        ax2.title.set_text("Oblique")

        plt.savefig("../Figures/SD/bias/sd_accuracy_card_obl_%s_%d_%d.png" % (subj, tmin, tmax))
        plt.clf()
    return card_bins, obl_bins

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

def split_half_orientation(subj):
    X, _, y, _ = ld.load_data(subj, data="epochs", n_classes=4, n_train=500, n_test=0, shuffle=True)
    

    kfold = KFold(n_splits=5)
    card_pred = []
    obl_pred = []

    for train, test in kfold.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        model = ml.LogisticSlidingModel(max_iter=4000, n_classes=4, k=20, C=0.1, l1_ratio=0.95)
        model.fit(X_train, y_train)

        X_card, y_card = X_test[y_test % 2 == 0], y_test[y_test % 2 == 0]
        X_obl, y_obl = X_test[y_test % 2 == 1], y_test[y_test % 2 == 1]

        card_pred.append(model.evaluate(X_card, y_card))
        obl_pred.append(model.evaluate(X_obl, y_obl))

    card_pred = np.mean(np.array(card_pred), axis=0)
    obl_pred = np.mean(np.array(obl_pred), axis=0)


    plt.plot(np.arange(0, 0.4, 0.025), card_pred, label="cardinal")
    plt.plot(np.arange(0, 0.4, 0.025), obl_pred, label='oblique')
    plt.ylim((0.2, 0.4))
    plt.legend()
    plt.savefig('../Figures/SD/split_half/split_half_or_%s.png' % subj)
    plt.clf()

    return card_pred, obl_pred

def analyze_probabilities(subj, show_plot=False):
    X, _, y, _ = ld.load_data(subj, data="epochs", n_train=500, n_test=0, n_classes=8)

    probabilities = np.zeros((16, 8))

    kfold = KFold(n_splits=5)
    for train, test in kfold.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        model = ml.LogisticSlidingModel(max_iter=4000, n_classes=8, k=20, C=0.085, l1_ratio=0.95)
        model.fit(X_train, y_train)
        pred = model.model.predict_proba(X_test)
        print(pred.shape)

        for i in range(16):
            for j in range(X_test.shape[0]):
                idx = int(y_test[j])
                new_probs = [pred[j, i, (idx+k) % 8] for k in range(-3, 5)]
                new_probs = np.array(new_probs)
                print(new_probs)
            

                probabilities[i, :] += new_probs
                

    probabilities = probabilities * (1/500)

    Xs = np.array([np.arange(0, 8) for _ in range(16)])
    Ys = np.array([np.ones(8) * i for i in range(16)])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xs, Ys, probabilities, cmap="coolwarm")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig("../Figures/SD/proba3D/proba3d_%s" % subj)
    if show_plot:
        plt.show()
    else:
        plt.clf()
    return probabilities

def analyze_probabilities_bias(subj, show_plot=False, plot2D=False, plot3D=False, n_bins=15, t=7):
    X, _, y, _ = ld.load_data(subj, data="epochs", n_train=500, n_test=0, n_classes=8)

    bin_sizes = np.zeros(n_bins)
    bins = [np.zeros(8) for i in range(n_bins)]
    bin_width = 180 // n_bins
    diffs = get_diffs(subj, shift=-1)

    kfold = KFold(n_splits=5)
    for train, test in kfold.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        split_diffs = diffs[test]

        model = ml.LogisticSlidingModel(max_iter=4000, n_classes=8, k=20, C=0.085, l1_ratio=0.95)
        model.fit(X_train, y_train)
        pred = model.model.predict_proba(X_test)
        print(pred.shape)

        for i in range(X_test.shape[0]):
            idx = int(y_test[i])
            bin_idx = split_diffs[i] // bin_width
            if bin_idx >= n_bins:
                bin_idx -= 1
            new_probs = np.array([pred[i, t, (idx+k) % 8] for k in range(-3, 5)])
            bins[bin_idx] += new_probs
            bin_sizes[bin_idx] += 1
                
    bins = np.array(bins)
    
    bin_accuracies = bins / bin_sizes[:, None]

    print(bin_accuracies.shape)
    labels = np.linspace(-90 + (bin_width/2), 90 - (bin_width/2), n_bins)

    if plot2D:
        plt.imshow(bin_accuracies.T)
        plt.ylabel("Class Prediction")
        plt.xlabel("Previous - Current Orientation")
        plt.colorbar()
        plt.savefig("../Figures/SD/proba2D/proba2d_bias_%s" % subj)
        if show_plot:
            plt.show()
        else:
            plt.clf()
    if plot3D:
        Xs = np.array([np.arange(0, 8) for _ in range(n_bins)])
        
        Ys = np.array([np.ones(8) * i for i in labels])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(Xs, Ys, bin_accuracies, cmap="coolwarm")
        ax.set_xlabel("Classes")
        ax.set_ylabel("Previous Orientation - Current Orientation")
        ax.set_zlabel("Class Probability")
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig("../Figures/SD/proba3D/proba3d_bias_%s" % subj)
        if show_plot:
            plt.show()
        else:
            plt.clf()
    return bins, bin_sizes














