import numpy as np
import mne
import sklearn
from scipy.io import loadmat

def load_behavioral_data(path):
    return loadmat(path)

def load_target_gabor(path):
    data = load_behavioral_data(path)
    return data["TargetGabor"][0]

def classify_target_gabors(path):
    gabor_lst = load_target_gabor(path)
    new_gabor_lst = []
    for gabor in gabor_lst:
        gabor = 1 if gabor > 0 else 0
        new_gabor_lst.append(gabor)
    return new_gabor_lst

