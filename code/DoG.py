import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model
from numpy import exp, loadtxt, pi, sqrt, std, mean, add
import scipy
from statistics import stdev 
from scipy import stats
from scipy.io import loadmat
import scipy.special as sps
import seaborn as sns

#from load_data import load_behavior
from file_lists import new_beh_lst, meg_subj_lst
#from serial_dependence import calc_relative_orientation

def calc_relative_orientation(x):
    if np.abs(x) > 90:
        x = x - (np.sign(x) * 180)
        return x
    return x

def load_behavior(subj):
    y_path = "../IEM-tutorial/Behavior/Sub%d_beh.mat" % new_beh_lst[subj]
    data = loadmat(y_path)
    pred = data["ResOrs"]
    actual = data["TgtOrs"]
    return pred.squeeze(), actual.squeeze()

def calc_rel_or_all(X):
    return np.array([calc_relative_orientation(i) for i in X])

def get_sd_data(subj, error_cutoff=25, or_cutoff=60, n=500):
    """
    Loads the predicted and actual orientations for the given subject

    returns: the relative orientation (previous - current stimulus),
             error (predicted - actual)
    """
    pred, actual = load_behavior(subj)
    actual = actual[~np.isnan(pred)]
    pred = pred[~np.isnan(pred)]
    length = len(pred)
    pred, actual = pred[:length].flatten(), actual[:length].flatten()

    rel_or = np.array([0] + [actual[i-1] - actual[i] for i in range(1, length)])
    rel_or = calc_rel_or_all(rel_or)
    
    

    error = calc_rel_or_all(pred - actual)

    rel_or = rel_or[np.abs(error) < error_cutoff]
    error = error[np.abs(error) < error_cutoff]

    error = error[np.abs(rel_or) < or_cutoff]
    rel_or = rel_or[np.abs(rel_or) < or_cutoff]

    """res, tgt = load_behavior(subj)
    res, tgt = res[0, :n], tgt[0, :n]
    rel_or = [0]
    first_err = calc_relative_orientation(res[0] - tgt[0])
    error = [first_err]
    for i in range(1, n):
        next_or = calc_relative_orientation(tgt[i-1] - tgt[i])
        next_er = calc_relative_orientation(res[i] - tgt[i])
        if np.abs(next_or) < or_cutoff and np.abs(next_er) < error_cutoff:
            rel_or += [next_or]
            error += [next_er]"""

    return rel_or, error


def gaussian(x, a=6, b=0.03):
    """
    1st order derivative gaussian
    """
    c = np.sqrt(2)/np.exp(-0.5)
    return (a * b * c * x * exp(-((b * x) ** 2)))

def init_gmodel(amax=40, amin=-40, bmax=60, bmin=-60):
    gmodel = Model(gaussian)
    gmodel.set_param_hint('a', max=amax, min=amin)
    gmodel.set_param_hint('b', max=bmax, min=bmin)
    return gmodel


def run_ptestv2(n_bootstraps=5000, n_permutations=100000, bootstrap_size=1000):
    subj_data = {}
    for subj in meg_subj_lst:
        next_or, next_error = get_sd_data(subj)
        subj_data[subj] = (next_or, next_error)

    fit_list = [[], []]
    a_list = np.zeros((len(meg_subj_lst), n_bootstraps))
    for i in range(n_bootstraps):
        if i % 500 == 0:
            print("Experimental Test %i" % i)

        for j, subj in enumerate(meg_subj_lst):
            subj_rel_or, subj_error = subj_data[subj]
            bootstrap_idx = np.random.choice(len(subj_rel_or), size=bootstrap_size, replace=True)
            subj_rel_or, subj_error = subj_rel_or[bootstrap_idx], subj_error[bootstrap_idx]
            gmodel = init_gmodel()
            result = gmodel.fit(subj_error, x=subj_rel_or, b=0.03)
            a_list[j, i] = result.params["a"]
            if i % 100 == 0:
                fit_list[0].extend(subj_rel_or)
                fit_list[1].extend(result.best_fit)


    perm_fit_list = [[], []]
    a_perm_list = np.zeros((len(meg_subj_lst), n_permutations))
    for i in range(n_permutations):
        if i % 1000 == 0:
            print("Permutation Test %i" % i)
        for j, subj in enumerate(meg_subj_lst):
            subj_rel_or, subj_error = subj_data[subj]
            bootstrap_idx = np.random.choice(len(subj_rel_or), size=bootstrap_size, replace=True)
            subj_rel_or, subj_error = subj_rel_or[bootstrap_idx], subj_error[bootstrap_idx]
            np.random.shuffle(subj_rel_or)
            gmodel = init_gmodel()
            result = gmodel.fit(subj_error, x=subj_rel_or, b=0.03)
            a_perm_list[j, i] = result.params["a"]
            if i % 1000 == 0:
                perm_fit_list[0].extend(subj_rel_or)
                perm_fit_list[1].extend(result.best_fit)


    a_list = a_list.flatten()
    a_perm_list = a_perm_list.flatten()
    p_value = 0

    for i in range(n_bootstraps):
        bootstrap_idx = np.random.choice(len(a_list), bootstrap_size, replace=True)
        boot_a_list = a_list[bootstrap_idx]

        bootstrap_idx = np.random.choice(len(a_perm_list), bootstrap_size, replace=True)
        boot_a_perm_list = a_perm_list[bootstrap_idx]

        if boot_a_perm_list.mean() >= boot_a_list.mean():
            p_value += 1
        
    p_value /= (n_bootstraps)

    rel_or, error = [], []
    for subj in meg_subj_lst:
        next_or, next_err = subj_data[subj]
        rel_or.extend(next_or)
        error.extend(next_err)

    rel_or, error = np.array(rel_or), np.array(error)

    gmodel = init_gmodel()
    result = gmodel.fit(error, x=rel_or, b=0.03)

    sorted_indices = np.argsort(rel_or)
    rel_or = rel_or[sorted_indices]




    print(p_value)
    print(a_list)
    print(a_perm_list)
    plt.figure(figsize=(9, 6))
    picked_scatter = np.random.choice(len(rel_or), 1000, replace=False)
    plt.scatter(rel_or[picked_scatter], error[picked_scatter], alpha=0.15)
    #sns.lineplot(fit_list[0], fit_list[1], color="r", label="Experimental")
    plt.plot(rel_or, result.best_fit[sorted_indices], color="red", linewidth=4, label="Experimental")
    sns.lineplot(perm_fit_list[0], perm_fit_list[1], color="g", label="Permutation", linewidth=4)
    plt.xlabel("Relative Orientation of Previous Trial")
    plt.ylabel("Error on Current Trial")
    plt.title("a={:.3f}      P={:.3f}".format(a_list.mean(), p_value))
    plt.savefig("../Figures/final_results/DoG/DoG_plot_v2.png", dpi=800)


def run_ptest(n_bootstraps=5000, n_permutations=100000, bootstrap_size=1000):
    rel_or, error = [], []
    for subj in meg_subj_lst:
        next_or, next_error = get_sd_data(subj)
        rel_or.extend(next_or)
        error.extend(next_error)

    rel_or, error = np.array(rel_or), np.array(error)
    mean_error = error - np.mean(error)

    # Perform bootstrap test
    a_list = []
    bootstrap_list = [[], []]
    for i in range(n_bootstraps):
        
        bootstrap_idx = np.random.choice(len(rel_or), size=bootstrap_size, replace=True)
        bootstrap_rel_or = rel_or[bootstrap_idx]
        bootstrap_error = mean_error[bootstrap_idx]
        gmodel = init_gmodel()
        result = gmodel.fit(bootstrap_error, x=bootstrap_rel_or, b=0.03)
        a_list.append(result.params['a'])
        if i % 500 == 0:
            print("Exp test: %d" % i)
            bootstrap_list[0].extend(bootstrap_rel_or)
            bootstrap_list[1].extend(result.best_fit)

    # Perform permutation test
    a_perm_list = []
    permutation_list = [[], []]
    for i in range(n_permutations):
        permutation_idx = np.random.choice(len(rel_or), size=len(rel_or), replace=False)
        permutation_rel_or = rel_or[permutation_idx]
        gmodel = init_gmodel()
        result = gmodel.fit(mean_error, x=permutation_rel_or, b=0.03)
        a_perm_list.append(result.params['a'])
        if i % 500 == 0:
            print("Perm test: %d" % i)
            permutation_list[0].extend(permutation_rel_or)
            permutation_list[1].extend(result.best_fit)

    a_list = np.array(a_list)
    a_perm_list = np.array(a_perm_list)

    p_num = 0
    for i in range(n_bootstraps):
        p_num += len(a_perm_list[a_perm_list > a_list[i]])

    p_num = p_num / (n_bootstraps * n_permutations)

    print("P Value: {:.3f}".format(p_num))
    print("Bootstrapped Average Amplitude: {:.3f}".format(a_list.mean()))
    print("Permuted Average Amplitude: {:.3f}".format(a_perm_list.mean()))
    plt.figure(figsize=(9, 6))
    plt.scatter(rel_or, error, alpha=0.25)
    sns.lineplot(bootstrap_list[0], bootstrap_list[1], color="r", label="Bootstrapped DoG", linewidth=3.5)
    sns.lineplot(permutation_list[0], permutation_list[1], color="g", label="Permutation DoG", linewidth=3.5)
    plt.xlabel("Relative Orientation of Previous Trial")
    plt.ylabel("Error on Current Trial")
    plt.title("a={:.3f}      P={:.3f}".format(a_list.mean(), p_num))
    plt.savefig("../Figures/final_results/DoG/DoG_plot.png")

def run_all_subj():
    rel_or, error = [], []
    for subj in meg_subj_lst:
        next_or, next_error = get_sd_data(subj)
        rel_or.extend(next_or)
        error.extend(next_error)

    rel_or, error = np.array(rel_or), np.array(error)
    mean_error = error - np.mean(error)

    print(error.shape)
    print(rel_or.shape)

    gmodel = init_gmodel()
    result = gmodel.fit(mean_error, x=rel_or, b=0.03)

    

    plt.figure(figsize=(12,8))
    plt.scatter(rel_or, error)

    sorted_indices = np.argsort(rel_or)
    rel_or = rel_or[sorted_indices]
    plt.plot(rel_or, result.best_fit[sorted_indices], color="red", linewidth=4)
    plt.xlabel("Previous Orientation - Current Orientation")
    plt.ylabel("Response Orientiaion - Presented Orientation")
    plt.xlim((-60, 60))
    plt.ylim((-40, 40))
    plt.savefig("../Figures/SD/DoG/DoG.png")
    plt.clf()

def run_subj(subj):
    rel_or, error = [], []
    next_or, next_error = get_sd_data(subj)
    rel_or.extend(next_or)
    error.extend(next_error)

    rel_or, error = np.array(rel_or), np.array(error)
    mean_error = error - np.mean(error)

    print(error.shape)
    print(rel_or.shape)

    gmodel = init_gmodel()
    result = gmodel.fit(mean_error, x=rel_or, b=0.03)

    

    plt.figure(figsize=(12,8))
    plt.scatter(rel_or, error)

    sorted_indices = np.argsort(rel_or)
    rel_or = rel_or[sorted_indices]
    plt.plot(rel_or, result.best_fit[sorted_indices], color="red", linewidth=4)
    plt.xlabel("Previous Orientation - Current Orientation")
    plt.ylabel("Response Orientiaion - Presented Orientation")
    plt.xlim((-60, 60))
    plt.ylim((-40, 40))
    plt.savefig("../Figures/SD/DoG/DoG_%s.png" % subj)
    plt.clf()

def main():
    run_ptestv2(n_bootstraps=5000, n_permutations=20000)

if __name__ == "__main__":
    main()



    