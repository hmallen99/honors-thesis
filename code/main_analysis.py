import mne
import re
import os
import numpy as np
import source_localization as srcl
import MEG_analysis as meg
import machine_learning as ml
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
                n_train=400, n_test=100):
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
                                                        n_classes=n_classes, use_off=True, data=data)
        model = ml.LogisticSlidingModel(max_iter=1500, n_classes=n_classes, k=20, C=0.1, l1_ratio=0.95)
        figure_label = "epochs"

    # Train Model with Source Estimate data
    if data == "stc":
        X_train, X_test, y_train, y_test = ld.load_data(behavior_subj, n_train=n_train, n_test=n_test,
                                                        n_classes=n_classes, use_off=True, data=data)
        model = ml.LogisticSlidingModel(max_iter=4000, n_classes=n_classes, k=1000, C=0.05, l1_ratio=0.95)
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
        model.get_features(behavior_subj, 9)
        results = model.evaluate(X_test, y_test)
        
    if data == "stc" or data == "epochs":
        ml.plot_results(np.linspace(0, 0.375, 16), results, figure_label, subj)

    return results

def analyze_sd(subj):
    kfold = KFold(n_splits=5)

    _, labels = ld.load_behavior(subj)
    labels = labels[0, :500]
    diffs = [0]
    for i in range(1, 500):
        ors_diff = np.abs(labels[i-1] - labels[i])
        diffs.append(ors_diff)

    accuracies, split_diffs = [], []
    X, _, y, _ = ld.load_data(subj, n_train=500, n_test=0)
    for train, test in kfold.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        model = ml.LogisticSlidingModel(max_iter=4000, n_classes=4, k=1000, C=0.05, l1_ratio=0.95)
        model.fit(X_train, y_train)
        accuracies.extend(model.evaluate(X_test, y_test))
        split_diffs.extend(diffs[test]) 

    return accuracies, split_diffs

def analyze_sd_all_subjects():
    accuracies = []
    diffs = []
    for subj in meg_subj_lst:
        accuracy, diff = analyze_sd(subj)
        accuracies.extend(accuracy)
        diffs.extend(diff)

    return

    

def run_all_subjects(data='stc', mode="cross_val", permutation_test=False, n_train=400, n_test=100):
    training_results = []
    for subject in meg_subj_lst:
        result = run_subject(subject, data=data, mode=mode, permutation_test=permutation_test,
                            n_train=n_train, n_test=n_test,)
        training_results.append(result)
    
    training_error = np.std(np.array(training_results), axis=0)
    training_results = np.array(training_results).mean(0)
    if permutation_test:
        ml.plot_results(np.linspace(0, 0.375, 16), training_results, mode + "_error_permutation", data + "_average", training_err=training_error)
    else:
        ml.plot_results(np.linspace(0, 0.375, 16), training_results, mode + "_error", data + "_average", training_err=training_error)


def main():
    run_all_subjects(data="epochs")
    run_all_subjects(data="epochs", permutation_test=True)
    return 0


if __name__ == "__main__":
    main()