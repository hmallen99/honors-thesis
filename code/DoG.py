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
from serial_dependence import calc_relative_orientation

def calc_rel_or_all(X):
    return np.array([calc_relative_orientation(i) for i in X])

def get_sd_data(subj, error_cutoff=40, or_cutoff=60):
    """
    Loads the predicted and actual orientations for the given subject

    returns: the relative orientation (previous - current stimulus),
             error (predicted - actual)
    """
    pred, actual = load_behavior(subj)
    pred, actual = pred[:, :500].flatten(), actual[:, :500].flatten()

    rel_or = -np.diff(pred, prepend=pred[0])
    rel_or = calc_rel_or_all(rel_or)
    

    error = calc_rel_or_all(pred - actual)
    rel_or = rel_or[np.abs(error) < error_cutoff]
    error = error[np.abs(error) < error_cutoff]

    error = error[np.abs(rel_or) < or_cutoff]
    rel_or = rel_or[np.abs(rel_or) < or_cutoff]

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

def main():
    rel_or, error = [], []
    for subj in new_beh_lst.keys():
        next_or, next_error = get_sd_data(subj)
        rel_or.extend(next_or)
        error.extend(next_error)

    rel_or, error = np.array(rel_or), np.array(error)
    error = error - np.mean(error)

    print(error.shape)
    print(rel_or.shape)

    gmodel = init_gmodel()
    result = gmodel.fit(error, x=rel_or, b=0.03)

    sorted_indices = np.argsort(rel_or)
    rel_or = rel_or[sorted_indices]

    plt.figure(figsize=(12,8))
    plt.scatter(rel_or, error)
    plt.plot(rel_or, result.best_fit[sorted_indices], color="red", linewidth=4)
    plt.xlabel("Previous Orientation - Current Orientation")
    plt.ylabel("Response Orientiaion - Presented Orientation")
    plt.savefig("../Figures/SD/DoG.png")
    plt.clf()


if __name__ == "__main__":
    main()



    