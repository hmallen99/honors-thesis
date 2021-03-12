import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import load_data as ld
import machine_learning as ml
from file_lists import meg_subj_lst, ch_picks
from serial_dependence import get_diffs
from scipy.io import loadmat



class InvertedEncoder(object):
    def __init__(self, n_ori_chans):
        self.n_ori_chans = n_ori_chans

    def make_basis_function(self, xx, mu):
        exp = self.n_ori_chans - (self.n_ori_chans % 2)
        return np.power((np.cos(xx - mu) * 180), exp)

    def make_basis_set(self):
        xx = np.linspace(1, 180, 180)
        basis_set = np.zeros((180, self.n_ori_chans))
        chan_center = np.linspace(180/self.n_ori_chans, 180, self.n_ori_chans)

        for cc in range(self.n_ori_chans):
            basis_set[:, cc] = self.make_basis_function(xx, chan_center[cc])

        return basis_set, chan_center

    def make_stimulus_mask(self, trng):
        stim_mask = np.zeros((trng.shape[0], 180))
        stim_mask[np.arange(trng.shape[0]), trng] = 1
        return stim_mask

    def cross_validate(self, trn, trng, tmin=0, tmax=16):
        trng = trng.flatten().astype(int) * (180 // self.n_ori_chans)
        self.basis_set, self.chan_center = self.make_basis_set()
        self.stim_mask = self.make_stimulus_mask(trng)
        self.trnX = self.stim_mask@self.basis_set

        trn_ou = np.unique(trng)
        trn_repnum = np.zeros(trng.shape[0])
        n_trials_per_orientation = np.zeros((len(trn_ou), 1))
        trng_flat = trng

        for ii in range(len(trn_ou)):
            num_idx = trng_flat==trn_ou[ii]
            n_trials = trng_flat[num_idx].shape[0]
            n_trials_per_orientation[ii] = n_trials
            trn_repnum[num_idx] = np.arange(n_trials)

        trn_repnum[trn_repnum > np.min(n_trials_per_orientation)] = np.nan

        trng_cv = trng_flat[~np.isnan(trn_repnum)]
        trn_cv = trn[~np.isnan(trn_repnum)]
        trnX_cv = self.trnX[~np.isnan(trn_repnum)]

        trn_cv_coeffs = np.zeros((trn_cv.shape[0], 2*trn_cv.shape[1], trn_cv.shape[2]))
        trn_cv_coeffs[:, :trn_cv.shape[1], :] = np.real(trn_cv)
        trn_cv_coeffs[:, trn_cv.shape[1]:, :] = np.imag(trn_cv)

        return_repnum = trn_repnum
        trn_repnum = trn_repnum[~np.isnan(trn_repnum)]

        chan_resp_cv_coeffs = np.zeros((trn_cv_coeffs[:].shape[0], self.chan_center.shape[0], trn_cv_coeffs[:].shape[2]))
        
        cv_iters = int(np.max(trn_repnum))
        
        
        for i in range(cv_iters):
            trn_idx = trn_repnum != i
            tst_idx = trn_repnum == i

            cur_trn = trn_cv_coeffs[trn_idx, :, :]
            cur_tst = trn_cv_coeffs[tst_idx, :, :]

            for j in range(trn_cv_coeffs.shape[2]):
                cur_trn_pt = cur_trn[:, :, j]
                cur_tst_pt = cur_tst[:, :, j]

                w_coeffs = np.linalg.lstsq(trnX_cv[trn_idx, :], cur_trn_pt)[0]
                
                wT_w_inv= np.linalg.inv(w_coeffs @ w_coeffs.T)
                wT_b = w_coeffs @ cur_tst_pt.T
                chan_resp_cv_coeffs[tst_idx, :, j] = wT_w_inv @ wT_b
                
        
        scores = np.zeros(chan_resp_cv_coeffs.shape[2])
        for i in range(chan_resp_cv_coeffs.shape[2]):
            preds = chan_resp_cv_coeffs[:, :, i].argmax(axis=1) * (180 // self.n_ori_chans)
            n_correct = trng_cv[preds.astype(int) == trng_cv].shape[0]
            n_total = trng_cv.shape[0]
            scores[i] = n_correct / n_total

        chan_resp_cv_coeffs_shift = np.zeros(chan_resp_cv_coeffs.shape)
        
        tar_idx = int(np.round(self.chan_center.shape[0] / 2))
        for i in range(self.n_ori_chans):
            idx = trng_cv == trn_ou[i]
            chan_resp_cv_coeffs_shift[idx, :, :] = np.roll(chan_resp_cv_coeffs[idx, :, :], tar_idx - i, axis=1)

        print(chan_resp_cv_coeffs_shift.shape)
        return scores, chan_resp_cv_coeffs_shift, return_repnum

def run_iem_subject(subj, n_classes=8, permutation_test=False):
    X, _, y, _ = ld.load_data(subj, n_train=500, n_test=0, n_classes=n_classes, data="epochs", shuffle=False, ch_picks=ch_picks)
    #X, y = load_eeg()
    if (permutation_test):
        shuffle_idx = np.random.choice(500, 500, replace=False)
        y = y[shuffle_idx]
    model = InvertedEncoder(n_classes)
    scores, _, _ = model.cross_validate(X, y)
    name = "IEM_cv"
    if permutation_test:
        name += "_permutation"
    ml.plot_results(np.arange(scores[:].shape[0]), scores, name, subj, ymin=0.075, ymax=0.175)
    return scores

def run_iem_3D(subj, n_classes=8, permutation_test=False, flat=False):
    X, _, y, _ = ld.load_data(subj, n_train=500, n_test=0, n_classes=n_classes, data="epochs", shuffle=False, ch_picks=ch_picks)
    if (permutation_test):
        shuffle_idx = np.random.choice(500, 500, replace=False)
        y = y[shuffle_idx]
    model = InvertedEncoder(n_classes)
    _, chan_resps, _ = model.cross_validate(X, y)
    mean_chan_resps = chan_resps.mean(axis=0)
    if flat:
        plt.imshow(mean_chan_resps, vmin=-1e14, vmax=1e14, cmap="autumn")
        plt.colorbar()
        plt.xlabel("timestep")
        plt.ylabel("channel response")
    else:
        Xs = np.array([np.arange(n_classes) for _ in range(16)])
        Ys = np.array([np.ones(n_classes) * i for i in np.arange(16)])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(Xs, Ys, mean_chan_resps.T, cmap="coolwarm")
        fig.colorbar(surf, shrink=0.5, aspect=5)

    name = "ml_results_iem_cv3d"
    if permutation_test:
        name += "_permutation"
    plt.savefig("../Figures/ML/%s_%s.png" % (name, subj))
    plt.clf()
    return mean_chan_resps

def iem_plot_timesteps(subj, n_classes=8, t=7, permutation_test=False):
    #X, _, y, _ = ld.load_data(subj, n_train=500, n_test=0, n_classes=n_classes, data="epochs", shuffle=False, ch_picks=ch_picks)
    X, y = load_eeg()
    if (permutation_test):
        shuffle_idx = np.random.choice(500, 500, replace=False)
        y = y[shuffle_idx]
    model = InvertedEncoder(n_classes)
    _, chan_resps, _ = model.cross_validate(X, y)
    #chan_resps = chan_resps[np.abs(chan_resps).max(axis=(1,2)) < 5e15]
    mean_chan_resps = chan_resps[:, :, t].mean(axis=0)
    plt.plot(mean_chan_resps)
    plt.savefig("../Figures/ML/IEM/bell_curve/bell_%s_%d.png" % (subj, t))
    plt.clf()

    print(chan_resps.shape)
    chan_resps_time_t = chan_resps[:, :, t]
    Xs = np.array([np.arange(n_classes) for _ in range(chan_resps_time_t[:].shape[0])])
    Ys = np.array([np.ones(n_classes) * i for i in np.arange(chan_resps_time_t[:].shape[0])])
    print(chan_resps_time_t.shape)
    print(Xs.shape)
    print(Ys.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xs, Ys, chan_resps_time_t, cmap="coolwarm")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig("../Figures/ML/IEM/bell_curve/bell_3d_%s_%d.png" % (subj, t))
    plt.clf()


def iem_serial_dependence(subj, n_classes=8, n_bins=15, t=7):
    diffs = get_diffs(subj, shift=-1)
    bin_sizes = np.zeros(n_bins)
    bins = [np.zeros(8) for i in range(n_bins)]
    bin_width = 180 // n_bins

    X, _, y, _ = ld.load_data(subj, n_train=500, n_test=0, n_classes=n_classes, data="epochs", shuffle=False, ch_picks=ch_picks)
    #X, y = load_eeg()

    model = InvertedEncoder(n_classes)
    _, chan_resps, repnum = model.cross_validate(X, y)

    diffs = diffs[~np.isnan(repnum)]
    for i in range(chan_resps[:].shape[0]):
        bin_idx = diffs[i] // bin_width
        if bin_idx >= n_bins:
            bin_idx -= 1
        bin_sizes[bin_idx] += 1
        bins[bin_idx] += chan_resps[i, :, t]

    bins = np.array(bins)
    bin_accuracies = bins / bin_sizes[:, None]


    plt.imshow(bin_accuracies.T)
    plt.ylabel("Class Prediction")
    plt.xlabel("Previous - Current Orientation")
    plt.colorbar()
    plt.savefig("../Figures/ML/IEM/proba2d_bias_%s" % subj)
    plt.clf()

    return

def load_eeg():
    eeg = loadmat("../Data/s08_EEG_ori_mini.mat")
    X = eeg["trn"]
    y = eeg["trng"].flatten()
    print(X.shape)
    print(y.shape)
    print(y)
    y = y // 20
    return X, y 

def main():
    score_lst = np.zeros(16)
    count = 0
    for subj in meg_subj_lst:
        scores = run_iem_subject(subj, n_classes=9)
        score_lst += scores
        count += 1
    
    score_lst /= count
    ml.plot_results(np.arange(16), score_lst, "IEM_cv", "all")
    #iem_plot_timesteps("eeg", n_classes=9)
    #run_iem_subject("AK", n_classes=9)



if __name__ == "__main__":
    main()