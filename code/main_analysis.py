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

meg_subj_lst = [
    "KA",
    "MF",
    #"MK",
    #"NNo",
    #"KO",
    "HHy",
    #"HO",
    "AK",
    "HN",
    "NN",
    "JL",
    "DI",
    "TE",
    "SoM",
    "VA",
    #"RS",
    "YMi",
]

aligned_dir = {
    "KA": "KA-aligned",
    "MF": "MF-aligned",
    "MK":  "MK-aligned",
    "NNo": "NNo-4",
    "KO": "KO-aligned",
    "HHy": "HHy-aligned",
    "HO": "HO-aligned",
    "AK": "AK-aligned",
    "HN": "HN-aligned",
    "NN": "NN-aligned",
    "JL": "JL-aligned",
    "DI": "DI-aligned",
    "SoM": "SoM-aligned",
    "TE": "TE-aligned",
    "RS": "RS-aligned",
    "VA": "VA-aligned",
    "YMi": "YMi-aligned",
}

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
                n_train=500, n_test=0, time_shift=0, use_off=True):
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
    n_classes = 4

    # Train Model with Epoch data
    if data == "epochs":
        X_train, X_test, y_train, y_test = ld.load_data(behavior_subj, n_train=n_train, n_test=n_test, 
                                                        n_classes=4, use_off=use_off, data=data,
                                                        time_shift=time_shift, mode="keras")
        #model = ml.LogisticSlidingModel(max_iter=1500, n_classes=n_classes, k=20, C=0.08, l1_ratio=0.95)
        model = ml.DenseSlidingModel(n_classes=4, n_epochs=5)
        figure_label = "epochs"

    # Train Model with Source Estimate data
    if data == "stc":
        X_train, X_test, y_train, y_test = ld.load_data(behavior_subj, n_train=n_train, n_test=n_test,
                                                        n_classes=n_classes, use_off=use_off, data=data,
                                                        time_shift=time_shift)
        model = ml.LogisticSlidingModel(max_iter=4000, n_classes=n_classes, k=1000, C=0.05, l1_ratio=0.95)
        #model = ml.DenseSlidingModel()
        figure_label = "source"

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
        
    if data == "stc" or data == "epochs":
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

def analyze_bias_all_subjects(tmin=0, tmax=16, n_bins=15, normalize=False, time_shift=-1, plot_individual=False):
    bins = [[] for i in range(n_bins)]
    for subj in meg_subj_lst:
        new_bins = sd.analyze_bias(subj, tmin, tmax, n_bins, normalize=normalize, time_shift=time_shift, plot=plot_individual)
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

def split_half_analysis_all():
    far_training_results = []
    close_training_results = []
    for subj in meg_subj_lst:
        close, far = sd.split_half_analysis(subj)

        close_training_results.append(close)
        far_training_results.append(far)

    close_training_results = np.array(close_training_results).mean(0)
    far_training_results = np.array(far_training_results).mean(0)
    
    plt.plot(np.arange(0, 0.4, 0.025), close_training_results, label="close")
    plt.plot(np.arange(0, 0.4, 0.025), far_training_results, label="far")
    plt.ylim((0.2, 0.4))
    plt.legend()
    plt.savefig('../Figures/SD/split_half/split_half_all.png')
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
    

def run_all_subjects(data='stc', mode="cross_val", permutation_test=False, n_train=500, n_test=0, time_shift=0, use_off=True):
    training_results = []
    for subject in meg_subj_lst:
        result = run_subject(subject, data=data, mode=mode, permutation_test=permutation_test,
                            n_train=n_train, n_test=n_test, time_shift=time_shift, use_off=use_off)
        training_results.append(result)
    
    training_error = np.std(np.array(training_results), axis=0)
    training_results = np.array(training_results).mean(0)
    if permutation_test:
        ml.plot_results(np.linspace(0, 0.375, 16), training_results, mode + "_error_permutation", data + "_average", training_err=training_error)
    else:
        ml.plot_results(np.linspace(0, 0.375, 16), training_results, mode + "_error", data + "_average", training_err=training_error)

    return training_results, training_error


def main():
    run_all_subjects(data="epochs", permutation_test=True, time_shift=0)

    """for i in [-1, -2, 1]:
        analyze_bias_all_subjects(tmin=6, tmax=9, time_shift=i)
        analyze_bias_all_subjects(tmin=7, tmax=8, time_shift=i)
        analyze_bias_all_subjects(tmin=0, tmax=16, time_shift=i)
        analyze_bias_all_subjects(tmin=9, tmax=11, time_shift=i)
        analyze_bias_all_subjects(tmin=13, tmax=16, time_shift=i)
        analyze_bias_all_subjects(tmin=3, tmax=5, time_shift=i)

        analyze_selectivity_all_subjects(tmin=6, tmax=9, time_shift=i)
        analyze_selectivity_all_subjects(tmin=7, tmax=8, time_shift=i)
        analyze_selectivity_all_subjects(tmin=0, tmax=16, time_shift=i)
        analyze_selectivity_all_subjects(tmin=9, tmax=11, time_shift=i)
        analyze_selectivity_all_subjects(tmin=13, tmax=16, time_shift=i)
        analyze_selectivity_all_subjects(tmin=3, tmax=5, time_shift=i)"""
    return 0


if __name__ == "__main__":
    main()