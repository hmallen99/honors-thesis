import numpy as np
from scipy.io import loadmat
import mne
import source_localization as srcl
import MEG_analysis as meg
import machine_learning as ml
from file_lists import new_beh_lst, meg_subj_lst, meg_locations, aligned_dir, behavior_lst

def load_y(subj, n=500, n_classes=2, use_off=True):
    y_path = "../../../../MEG/Behaviour/Sub%d_beh.mat" % new_beh_lst[subj]
    data = loadmat(y_path)["TgtOrs"]
    if n_classes == -1:
        return data.flatten().astype(float)[:n]
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

def generate_epoch_X(epochs, n=500, ch_picks=[]):
    epochs.load_data().resample(40)
    meg_epochs = []
    if len(ch_picks) == 0:
        meg_epochs = epochs.copy().pick_types(meg=True, eeg=False).crop(0, 0.4, False)
    else:
        meg_epochs = epochs.copy().pick_channels(ch_picks).crop(0, 0.4, False)
    X = meg_epochs.get_data()
    return X[:n]

def generate_wave_X(epochs, n=500):
    freqs = np.arange(5, 41, 5)
    n_cycles = freqs / 5
    print(freqs)
    epochs.load_data().resample(40)
    meg_epochs = epochs.copy().pick_types(meg=True, eeg=False).crop(0, 1, False)
    meg_epochs = meg_epochs.get_data()[:n]
    meg_wavelets = mne.time_frequency.tfr_array_morlet(meg_epochs, sfreq=80, freqs=freqs, n_cycles=n_cycles, output="power")
    return np.expand_dims(np.array(meg_wavelets), -1)

def shift_time_step(time_shift, X, y, n):
    if time_shift == -1:
        X[:-1] = X[1:]
        idx = np.random.choice(n)
        X[-1] = X[idx]
        y[-1] = y[idx]
        return X, y
    elif time_shift == -2:
        X[:-2] = X[2:]
        idx = np.random.choice(n)
        X[-1] = X[idx]
        y[-1] = y[idx]

        idx = np.random.choice(n)
        X[-2] = X[idx]
        y[-2] = y[idx]
        return X, y
    elif time_shift == 1:
        y[:-1] = y[1:]
        idx = np.random.choice(n)
        X[-1] = X[idx]
        y[-1] = y[idx]
        return X, y
    else:
        return X, y


def get_stc_data(subj, stc_epoch, n_train=400, n_test=100, n_classes=4, use_off=True, mode="sklearn", shuffle=False, time_shift=0):
    y = load_y(subj, n=n_train+n_test, n_classes=n_classes, use_off=use_off)
    X = generate_stc_X(stc_epoch, n=n_train+n_test, mode=mode)

    if time_shift != 0:
        X, y = shift_time_step(time_shift, X, y, n_train + n_test)

    if shuffle:
        idxs = np.arange(y.shape[0])
        np.random.shuffle(idxs)
        y = y[idxs]
        X = X[idxs]

    X_train, X_test = X[:n_train], X[n_train:n_train+n_test]
    y_train, y_test = y[:n_train], y[n_train:n_train+n_test]
    return X_train, X_test, y_train, y_test

def get_epoch_data(subj, epochs, n_train=400, n_test=100, n_classes=4, use_off=True, shuffle=False, time_shift=0, ch_picks=[]):
    y = load_y(subj, n=n_train+n_test, n_classes=n_classes, use_off=use_off)
    X = generate_epoch_X(epochs, n=n_train+n_test, ch_picks=ch_picks)

    if time_shift != 0:
        X, y = shift_time_step(time_shift, X, y, n_train + n_test)

    if shuffle:
        idxs = np.arange(y.shape[0])
        np.random.shuffle(idxs)
        y = y[idxs]
        X = X[idxs]

    X_train, X_test = X[:n_train], X[n_train:n_train+n_test]
    y_train, y_test = y[:n_train], y[n_train:n_train+n_test]
    return X_train, X_test, y_train, y_test

def get_wave_data(subj, epochs, n_train=400, n_test=100, n_classes=4, use_off=True, shuffle=False, time_shift=0):
    y = load_y(subj, n=n_train+n_test, n_classes=n_classes, use_off=use_off)
    X = generate_wave_X(epochs, n=n_train+n_test)

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


def load_data(behavior_subj, n_train=400, n_test=100, n_classes=4, use_off=True, shuffle=False, data="stc", mode="sklearn", time_shift=0, ch_picks=[]):
    folder_dict = meg.get_folder_dict()
    subj = aligned_dir[behavior_subj]
    meg_dir = meg.meg_locations[behavior_subj]
    source_localization_dir = "/usr/local/freesurfer/subjects"

    # Collect Data
    epochs, evoked = meg.get_processed_meg_data(subj, folder_dict, meg_dir)

    bad = []
    for i in range(len(epochs.events)):
        if epochs.events[i][1] != 0:
            bad += [i]

    epochs.drop(bad)

    if data == "epochs":
        return get_epoch_data(behavior_subj, epochs, n_train=n_train, n_test=n_test, n_classes=n_classes, use_off=use_off, shuffle=shuffle, time_shift=time_shift, ch_picks=ch_picks)

    if data == "stc":
        src, bem = srcl.get_processed_mri_data(subj, source_localization_dir)
        cov = mne.compute_covariance(epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=False)
        fwd = srcl.make_forward_sol(evoked, src, bem, "%s/%s-trans.fif" % (meg_dir, subj))

        inv_op_epoch = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, loose=0.2, depth=0.8)
        stc_epoch = mne.minimum_norm.apply_inverse_epochs(epochs, inv_op_epoch, 0.11, return_generator=True)

        return get_stc_data(behavior_subj, stc_epoch, n_train=n_train, n_test=n_test, n_classes=n_classes, use_off=use_off, shuffle=shuffle, mode=mode, time_shift=time_shift)

    if data == "wave":
        return get_wave_data(behavior_subj, epochs, n_train=n_train, n_test=n_test, n_classes=n_classes, use_off=use_off, shuffle=shuffle, time_shift=time_shift)

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