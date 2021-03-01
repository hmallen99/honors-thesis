import mne
import re
import os
import numpy as np
import source_localization as srcl
import MEG_analysis as meg
import machine_learning as ml
import serial_dependence as sd
import load_data as ld
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from load_data import meg_subj_lst, aligned_dir


def save_main_figs(subj):
    erf_fig = plt.imread("../Figures/residuals/%s_residual_erf.png" % aligned_dir[subj])
    behavior_fig = plt.imread("../Figures/Behavior/%s_behavior.png" % subj)
    accuracy_fig = plt.imread("../Figures/ML/ml_results_source_cross_val_%s.png" % aligned_dir[subj])
    
    fig = plt.figure(figsize=(24, 8))

    ax_erf = fig.add_subplot(1, 3, 1)
    ax_erf.imshow(erf_fig)

    ax_beh = fig.add_subplot(1, 3, 2)
    ax_beh.imshow(behavior_fig)

    ax_acc = fig.add_subplot(1, 3, 3)
    ax_acc.imshow(accuracy_fig)

    plt.savefig("../Figures/combined/%s_combined.png" % aligned_dir[subj], dpi=700)
    plt.clf()


def run_subject(behavior_subj, data="stc", mode="cross_val", permutation_test=False,
                n_train=500, n_test=0, time_shift=0, use_off=True, n_classes=4, model_data="sklearn",
                shuffle=False):
    """
    Runs the full ML pipeline for behavior_subj

    Returns evaluation or cross validation scores for the subject
    """
    subj = aligned_dir[behavior_subj]
    results = []
    model = None
    X_train, X_test = [], []
    y_train, y_test = [], []
    figure_label = "default"

    # Train Model with Epoch data
    if data == "epochs":
        X_train, X_test, y_train, y_test = ld.load_data(behavior_subj, n_train=n_train, n_test=n_test, 
                                                        n_classes=n_classes, use_off=use_off, data=data,
                                                        time_shift=time_shift, mode=model_data, shuffle=shuffle)
        if model_data == "sklearn":
            model = ml.LogisticSlidingModel(max_iter=1500, n_classes=n_classes, k=20, C=0.08, l1_ratio=0.95)
        elif model_data == "keras":
            #model = ml.GaborSlidingModel()
            model = ml.LogisticRNNModel(n_classes=n_classes)
        figure_label = "epochs"

    # Train Model with Source Estimate data
    if data == "stc":
        X_train, X_test, y_train, y_test = ld.load_data(behavior_subj, n_train=n_train, n_test=n_test,
                                                        n_classes=n_classes, use_off=use_off, data=data,
                                                        time_shift=time_shift, shuffle=shuffle)
        model = ml.LogisticSlidingModel(max_iter=4000, n_classes=n_classes, k=1000, C=0.05, l1_ratio=0.95)
        #model = ml.DenseSlidingModel()
        figure_label = "source"


    if data == "wave":
        X_train, X_test, y_train, y_test = ld.load_data(behavior_subj, n_train=n_train, n_test=n_test,
                                                        n_classes=n_classes, use_off=use_off, data=data,
                                                        time_shift=time_shift, shuffle=shuffle)
        
        model = ml.CNNSlidingModel(input_shape=X_train.shape[1:], n_classes=n_classes)
        figure_label = "wave"

    if permutation_test:
        figure_label += "_permutation"
        np.random.shuffle(y_train)
        np.random.shuffle(y_test)

    if mode == "cross_val":
        figure_label += "_cross_val"
        results = model.cross_validate(X_train, y_train) 
    elif mode == "evaluate":
        figure_label += "_test"
        model.fit(X_train, y_train)
        if data == "stc":
            vertices = ld.get_vertices(behavior_subj)
            model.plot_weights_stc(subj, vertices)
        elif data == "epochs":
            epochs = ld.get_epochs(behavior_subj)
            model.plot_weights_epochs(subj, epochs)
        results = model.evaluate(X_test, y_test)
        
    if data == "stc" or data == "epochs" or data == "wave":
        ml.plot_results(np.linspace(0, 0.375, 16), results, figure_label, subj)

    return results

def analyze_selectivity_all_subjects(tmin=0, tmax=16, n_bins=15, time_shift=-1):
    bins = {}
    for i in range(n_bins):
        bins[i] = []

    for subj in meg_subj_lst:
        subj_bins = sd.analyze_selectivity(subj, tmin=tmin, tmax=tmax, n_bins=n_bins, time_shift=time_shift)
        for i in range(n_bins):
            bins[i].extend(subj_bins[i])

    bin_accuracies = [np.mean(np.array(bins[i])) for i in range(n_bins)]
    #bin_stds = [np.std(np.array(bins[i])) for i in range(n_bins)]
    bin_sizes = [len(bins[i]) for i in range(n_bins)]

    bin_width = 180 // n_bins
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(n_bins), bin_accuracies)

    for i, s in enumerate(bin_sizes):
        plt.text(i, bin_accuracies[i], str(s), color="blue", fontweight="bold")

    plt.xticks(ticks=np.arange(n_bins), labels=np.linspace(-90 + (bin_width/2), 90 - (bin_width/2), n_bins))
    plt.savefig("../Figures/SD/selectivity/sd_accuracy_all_%d_%d_%d.png" % (tmin, tmax, time_shift))
    plt.clf()
    return

def analyze_bias_all_subjects(tmin=0, tmax=16, n_bins=15, n_classes=4, time_shift=-1, plot_individual=False):
    bins = [[] for i in range(n_bins)]
    for subj in meg_subj_lst:
        new_bins = sd.analyze_bias(subj, tmin, tmax, n_bins, n_classes=n_classes, time_shift=time_shift, plot=plot_individual)
        for i in range(n_bins):
            bins[i].extend(new_bins[i])

    bin_width = 180 // n_bins
    bin_accuracies = [np.mean(np.array(bins[i])) for i in range(n_bins)]
    #bin_stds = [np.std(np.array(bins[i])) for i in range(n_bins)]
    bin_sizes = [len(bins[i]) for i in range(n_bins)]

    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(n_bins), bin_accuracies)
    for i, s in enumerate(bin_sizes):
        plt.text(i+.05, bin_accuracies[i] + 0.01, str(s), color="blue", fontweight="bold")
    labels = np.linspace(-90 + (bin_width/2), 90 - (bin_width/2), n_bins)
    plt.xticks(ticks=np.arange(n_bins), labels=labels)
    plt.savefig("../Figures/SD/bias/sd_accuracy_all_%d_%d_%d.png" % (tmin, tmax, time_shift))
    plt.clf()
    return

def analyze_bias_card_obl_all_subjects(tmin=0, tmax=16, n_bins=15, n_classes=4, time_shift=-1, plot_individual=False):
    card_bins = [[] for i in range(n_bins)]
    obl_bins = [[] for i in range(n_bins)]
    for subj in meg_subj_lst:
        new_card_bins, new_obl_bins = sd.analyze_bias_card_obl(subj, tmin, tmax, n_bins, time_shift=time_shift, plot=plot_individual)
        for i in range(n_bins):
            card_bins[i].extend(new_card_bins[i])
            obl_bins[i].extend(new_obl_bins[i])

    bin_width = 180 // n_bins
    card_bin_accuracies = [np.mean(np.array(card_bins[i])) for i in range(n_bins)]
    obl_bin_accuracies = [np.mean(np.array(obl_bins[i])) for i in range(n_bins)]
    #bin_stds = [np.std(np.array(bins[i])) for i in range(n_bins)]
    #bin_sizes = [len(card_bins[i]) for i in range(n_bins)]

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

    plt.savefig("../Figures/SD/bias/sd_accuracy_all_card_obl_%d_%d_%d.png" % (tmin, tmax, time_shift))
    plt.clf()
    return

def split_half_analysis_all(mode="or_diff"):
    func = lambda x: np.zeros(16)
    labels = ["class1", "class2"]

    if mode == "or_diff":
        func = sd.split_half_analysis
        labels = ["close", "far"]
    elif mode == "or_class":
        func = sd.split_half_orientation
        labels = ["cardinal", "oblique"]

    class2_training_results = []
    class1_training_results = []
    for subj in meg_subj_lst:
        class1, class2 = func(subj)

        class1_training_results.append(class1)
        class2_training_results.append(class2)

    class1_training_results = np.array(class1_training_results).mean(0)
    class2_training_results = np.array(class2_training_results).mean(0)
    
    plt.plot(np.arange(0, 0.4, 0.025), class1_training_results, label=labels[0])
    plt.plot(np.arange(0, 0.4, 0.025), class2_training_results, label=labels[1])
    plt.ylim((0.2, 0.4))
    plt.legend()
    plt.savefig('../Figures/SD/split_half/split_half_all_%s.png' % mode)
    plt.clf()

def analyze_serial_dependence_all():
    errors = []
    rel_ors = []
    for subj in meg_subj_lst:
        rel, err = sd.analyze_serial_dependence(subj)
        errors.extend(err)
        rel_ors.extend(rel)

    bins = [[] for i in range(12)]

    for i in range(len(errors)):
        bin_idx = (rel_ors[i] + 60) // 10
        if bin_idx == 12:
            bin_idx = 11
        bins[bin_idx] += [errors[i]]


    means = [np.mean(np.array(bins[i])) for i in range(12)]
    plt.plot(np.arange(-55, 56, 10), means, color='r')

    plt.scatter(rel_ors, errors)
    plt.xlim((-60, 60))
    plt.ylim((-40, 40))
    plt.savefig("../Figures/SD/subj_sd/all_sd.png")
    plt.clf()

def analyze_probabilities_all():
    probabilities = np.zeros((16, 8))
    for subj in meg_subj_lst:
        probabilities += sd.analyze_probabilities(subj)

    probabilities = probabilities / len(meg_subj_lst)
    Xs = np.array([np.arange(0, 8) for _ in range(16)])
    Ys = np.array([np.ones(8) * i for i in range(16)])

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xs, Ys, probabilities, cmap="coolwarm")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig("../Figures/SD/proba3D/proba3d_all")
    plt.show()   

def run_all_subjects(data='stc', mode="cross_val", permutation_test=False, n_train=500, n_test=0, time_shift=0, use_off=True, 
                     n_classes=4, model_data="sklearn", shuffle=False):
    training_results = []
    for subject in meg_subj_lst:
        result = run_subject(subject, data=data, mode=mode, permutation_test=permutation_test,
                            n_train=n_train, n_test=n_test, time_shift=time_shift, use_off=use_off, 
                            n_classes=n_classes, model_data=model_data, shuffle=shuffle)
        training_results.append(result)
    
    training_error = np.std(np.array(training_results), axis=0)
    training_results = np.array(training_results).mean(0)
    if permutation_test:
        ml.plot_results(np.linspace(0, 0.375, 16), training_results, mode + "_error_permutation", data + "_average", training_err=training_error)
    else:
        ml.plot_results(np.linspace(0, 0.375, 16), training_results, mode + "_error", data + "_average", training_err=training_error)

    return training_results, training_error

def analyze_probabilities_bias_all(n_bins=15, t=7, show_plot=False, plot2D=True, plot3D=False):
    bins, bin_sizes = np.zeros((n_bins, 8)), np.zeros(n_bins)
    for subj in meg_subj_lst:
        new_bins, new_bin_sizes = sd.analyze_probabilities_bias(subj, n_bins=n_bins, plot2D=False, t=t)
        bins += new_bins
        bin_sizes += new_bin_sizes

    bin_accuracies = bins / bin_sizes[:, None]
    bin_width = 180 // n_bins

    Xs = np.array([np.arange(0, 8) for _ in range(n_bins)])
    labels = np.linspace(-90 + (bin_width/2), 90 - (bin_width/2), n_bins)
    Ys = np.array([np.ones(8) * i for i in labels])

    if plot2D:
        plt.imshow(bin_accuracies.T)
        plt.colorbar()
        plt.ylabel("Class Prediction")
        plt.xlabel("Previous - Current Orientation")
        #plt.yticks(Y)
        plt.savefig("../Figures/SD/proba2D/proba2d_bias_all_%d" % t)
        if show_plot:
            plt.show()
        else:
            plt.clf()
    if plot3D:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(Xs, Ys, bin_accuracies, cmap="coolwarm")
        ax.set_xlabel("Classes")
        ax.set_ylabel("Previous Orientation - Current Orientation")
        ax.set_zlabel("Class Probability")
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig("../Figures/SD/proba3D/proba3d_bias_all_%d" % t)
        if show_plot:
            plt.show()
        else:
            plt.clf()


def main():
    run_all_subjects(data="wave", permutation_test=False, n_classes=4, model_data="sklearn", shuffle=True)
    #run_all_subjects(data="wave", permutation_test=True, n_classes=4, model_data="sklearn", shuffle=True)
    #for i in range(0, 16):
    #    analyze_probabilities_bias_all(t=i)
    #analyze_probabilities_bias_all()
    #analyze_bias_card_obl_all_subjects(tmin=6, tmax=9, n_classes=8, time_shift=-1, plot_individual=True)
    return 0


if __name__ == "__main__":
    main()