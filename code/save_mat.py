import load_data as ld
import numpy as np
from file_lists import meg_subj_lst, ch_picks
from scipy.io import savemat, loadmat

for subj in meg_subj_lst:
    X, _, y, _ = ld.load_data(subj, n_train=500, n_test=0, n_classes=8, data="epochs", shuffle=False, ch_picks=[])

    y = y * 20
    ts = np.linspace(0, 0.4, 16)
    mat_dict = {
        "trng" : y,
        "trn" : X,
        "ts" : ts
    }
    savemat("../Data/mat_epochs/%s_epochs_8classes_all.mat" % subj, mat_dict)