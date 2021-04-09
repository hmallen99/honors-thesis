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
import scipy.special as sps

from load_data import load_behavior, new_beh_lst
from main_analysis import meg_subj_lst
from serial_dependence import calc_relative_orientation

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
    pred, actual = pred[:, :500].flatten(), actual[:, :500].flatten()

    rel_or = np.array([0] + [actual[i-1] - actual[i] for i in range(1, 500)])
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
    for i in range(n_bootstraps):
        bootstrap_idx = np.random.choice(len(rel_or), size=bootstrap_size, replace=True)
        bootstrap_rel_or = rel_or[bootstrap_idx]
        bootstrap_error = mean_error[bootstrap_idx]
        gmodel = init_gmodel()
        result = gmodel.fit(bootstrap_error, x=bootstrap_rel_or, b=0.03)
        a_list.append(result.params['a'])

    # Perform permutation test
    a_perm_list = []
    for i in range(n_permutations):
        permutation_idx = np.random.choice(len(rel_or), size=len(rel_or), replace=False)
        permutation_rel_or = rel_or[permutation_idx]
        gmodel = init_gmodel()
        result = gmodel.fit(mean_error, x=permutation_rel_or, b=0.03)
        a_perm_list.append(result.params['a'])

    a_list = np.array(a_list)
    a_perm_list = np.array(a_perm_list)

    p_num = 0
    for i in range(n_bootstraps):
        p_num += len(a_perm_list[a_perm_list > a_list[i]])

    p_num = p_num / (n_bootstraps * n_permutations)

    print(p_num)


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
    #for subj in meg_subj_lst:
        #run_subj(subj)

    #run_all_subj()
    run_ptest()

if __name__ == "__main__":
    main()



    