import numpy as np
from scipy.io import loadmat
import mne
import source_localization as srcl
import MEG_analysis as meg
import machine_learning as ml

new_beh_lst = {
    "KA": 1,
    "MF": 7,
    "MK": 12,
    "NNo": 19,
    "KO": 4,
    "HHy": 18,
    "HO": 6,
    "AK": 5,
    "HN": 21,
    "NN": 3,
    "JL": 9,
    "DI": 16,
    "SoM": 2,
    "TE": 17,
}

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
}

def load_y(subj, n=500, n_classes=2, use_off=True):
    y_path = "../../../../MEG/Behaviour/Sub%d_beh.mat" % new_beh_lst[subj]
    data = loadmat(y_path)["TgtOrs"]
    offset = 0
    if use_off:
        offset = 180 / (n_classes * 2)
    ors = (data + 90 + offset) % 180
    ors = np.floor(ors / (180 / n_classes))
    ors = np.minimum(ors, n_classes-1)[0]
    return ors[:n]

def load_behavior(subj):
    y_path = "../../../../MEG/Behaviour/Sub%d_beh.mat" % new_beh_lst[subj]
    data = loadmat(y_path)
    pred = data["ResOrs"]
    actual = data["TgtOrs"]
    return pred, actual

def generate_stc_X(stc_epoch, n=500, mode="sklearn"):
    X = []
    if mode == "sklearn":
        X = np.array([next(stc_epoch).crop(0, 0.4).bin(0.025).data for i in range(n)])
    elif mode == "keras":
        X = np.einsum('ikj->ijk', np.array([next(stc_epoch).crop(0, 0.4, False).bin(0.025).data for i in range(n)]))
    X_return = X[:n]
    del stc_epoch
    del X
    return X_return

def generate_epoch_X(epochs, n=500):
    epochs.load_data().resample(40)
    meg_epochs = epochs.copy().pick_types(meg=True, eeg=False).crop(0, 0.4, False)
    X = meg_epochs.get_data()
    return X[:n]


def get_stc_data(subj, stc_epoch, n_train=400, n_test=100, n_classes=4, use_off=True, mode="sklearn", shuffle=False, previous=False):
    y = load_y(subj, n=n_train+n_test, n_classes=n_classes, use_off=use_off)
    X = generate_stc_X(stc_epoch, n=n_train+n_test, mode=mode)

    if previous:
        X[:-1] = X[1:]
        idx = np.random.choice(n_train + n_test)
        X[-1] = X[idx]
        y[-1] = y[idx]

    if shuffle:
        idxs = np.arange(y.shape[0])
        np.random.shuffle(idxs)
        y = y[idxs]
        X = X[idxs]

    X_train, X_test = X[:n_train], X[n_train:n_train+n_test]
    y_train, y_test = y[:n_train], y[n_train:n_train+n_test]
    return X_train, X_test, y_train, y_test

def get_epoch_data(subj, epochs, n_train=400, n_test=100, n_classes=4, use_off=True, shuffle=False, previous=False):
    y = load_y(subj, n=n_train+n_test, n_classes=n_classes, use_off=use_off)
    X = generate_epoch_X(epochs, n=n_train+n_test)

    if previous:
        X[:-1] = X[1:]
        idx = np.random.choice(n_train + n_test)
        X[-1] = X[idx]
        y[-1] = y[idx]

    if shuffle:
        idxs = np.arange(y.shape[0])
        np.random.shuffle(idxs)
        y = y[idxs]
        X = X[idxs]

    X_train, X_test = X[:n_train], X[n_train:n_train+n_test]
    y_train, y_test = y[:n_train], y[n_train:n_train+n_test]
    return X_train, X_test, y_train, y_test


def get_vertices(behavior_subj):
    folder_dict = meg.get_folder_dict()
    subj = aligned_dir[behavior_subj]
    meg_dir = meg.meg_locations[behavior_subj]
    source_localization_dir = "/usr/local/freesurfer/subjects"

    # Collect Data
    epochs, evoked = meg.get_processed_meg_data(subj, folder_dict, meg_dir)

    src, bem = srcl.get_processed_mri_data(subj, source_localization_dir)
    cov = mne.compute_covariance(epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=False)
    fwd = srcl.make_forward_sol(evoked, src, bem, "%s/%s-trans.fif" % (meg_dir, subj))

    inv_op_epoch = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, loose=0.2, depth=0.8)
    stc_epoch = mne.minimum_norm.apply_inverse_epochs(epochs, inv_op_epoch, 0.11, return_generator=True)

    stc = next(stc_epoch)

    return stc.vertices

def get_epochs(behavior_subj):
    folder_dict = meg.get_folder_dict()
    subj = aligned_dir[behavior_subj]
    meg_dir = meg.meg_locations[behavior_subj]

    # Collect Data
    epochs, _ = meg.get_processed_meg_data(subj, folder_dict, meg_dir)

    bad = []
    for i in range(len(epochs.events)):
        if epochs.events[i][1] != 0:
            bad += [i]

    epochs.drop(bad)
    epochs.load_data().resample(40)
    meg_epochs = epochs.copy().pick_types(meg=True, eeg=False).crop(0, 0.4, False)
    return meg_epochs


def load_data(behavior_subj, n_train=400, n_test=100, n_classes=4, use_off=True, shuffle=False, data="stc", mode="sklearn", previous=False):
    folder_dict = meg.get_folder_dict()
    subj = aligned_dir[behavior_subj]
    meg_dir = meg.meg_locations[behavior_subj]
    source_localization_dir = "/usr/local/freesurfer/subjects"
    n_classes = 4

    # Collect Data
    epochs, evoked = meg.get_processed_meg_data(subj, folder_dict, meg_dir)

    bad = []
    for i in range(len(epochs.events)):
        if epochs.events[i][1] != 0:
            bad += [i]

    epochs.drop(bad)

    if data == "epochs":
        return get_epoch_data(behavior_subj, epochs, n_train=n_train, n_test=n_test, n_classes=n_classes, use_off=use_off, previous=previous)

    if data == "stc":
        src, bem = srcl.get_processed_mri_data(subj, source_localization_dir)
        cov = mne.compute_covariance(epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=False)
        fwd = srcl.make_forward_sol(evoked, src, bem, "%s/%s-trans.fif" % (meg_dir, subj))

        inv_op_epoch = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, loose=0.2, depth=0.8)
        stc_epoch = mne.minimum_norm.apply_inverse_epochs(epochs, inv_op_epoch, 0.11, return_generator=True)

        return get_stc_data(behavior_subj, stc_epoch, n_train=n_train, n_test=n_test, n_classes=n_classes, use_off=use_off, mode=mode, previous=previous)

    return None


def save_evoked_fig_plots(stc_fsaverage, subj, residual):
    residual.plot_topo(title='Residual Plot', show=False).savefig('../Figures/residuals/%s_residual_erf.png' % subj, dpi=500)
    #srcl.save_movie(stc_fsaverage, subj)

    #for i in [0, 0.1, 0.2, 0.3]:
        #srcl.plot_source(stc_fsaverage, subject=subj, initial_time=i, views="dorsal", hemi="both")
        #srcl.plot_source(stc_fsaverage, subject=subj, initial_time=i, views="caudal", hemi="both")
        #srcl.plot_source(stc_fsaverage, subject=subj, initial_time=i, views="lateral", hemi="rh")
        #srcl.plot_source(stc_fsaverage, subject=subj, initial_time=i, views="lateral", hemi="lh")

def save_evoked_figs(behavior_subj):
    folder_dict = meg.get_folder_dict()
    subj = aligned_dir[behavior_subj]
    meg_dir = meg.meg_locations[behavior_subj]
    source_localization_dir = "/usr/local/freesurfer/subjects"

    epochs, evoked = meg.get_processed_meg_data(subj, folder_dict, meg_dir)
    src, bem = srcl.get_processed_mri_data(subj, source_localization_dir)
    cov = mne.compute_covariance(epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=False)
    fwd = srcl.make_forward_sol(evoked, src, bem, "%s/%s-trans.fif" % (meg_dir, subj))

    inv_op = srcl.make_inverse_operator(evoked, fwd, cov)
    stc, residual = srcl.apply_inverse(evoked, inv_op)
    stc_fsaverage = srcl.morph_to_fsaverage(stc, subj)
    save_evoked_fig_plots(stc_fsaverage, subj, residual)