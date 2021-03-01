import numpy as np
import matplotlib.pyplot as plt
import load_data as ld
import machine_learning as ml



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

        trn_repnum = trn_repnum[~np.isnan(trn_repnum)]

        chan_resp_cv_coeffs = np.zeros((trn_cv_coeffs[:].shape[0], self.chan_center.shape[0], trn_cv_coeffs[:].shape[2]))
        
        cv_iters = int(np.max(trn_repnum))
        
        
        for i in range(cv_iters):
            trn_idx = trn_repnum != i
            tst_idx = trn_repnum == i

            cur_trn = trn_cv_coeffs[trn_idx, :, :]
            cur_tst = trn_cv_coeffs[tst_idx, :, :]

            for j in range(16):
                cur_trn_pt = cur_trn[:, :, j]
                cur_tst_pt = cur_tst[:, :, j]

                w_coeffs = np.linalg.lstsq(trnX_cv[trn_idx, :], cur_trn_pt)[0]
                
                wT_w_inv= np.linalg.inv(w_coeffs @ w_coeffs.T)
                wT_b = w_coeffs @ cur_tst_pt.T
                chan_resp_cv_coeffs[tst_idx, :, j] = wT_w_inv @ wT_b
                

        scores = np.zeros(16)
        for i in range(16):
            preds = chan_resp_cv_coeffs[:, :, i].argmax(axis=1) * (180 // self.n_ori_chans)
            n_correct = trng_cv[preds.astype(int) == trng_cv].shape[0]
            n_total = trng_cv.shape[0]
            scores[i] = n_correct / n_total

        return scores

def run_iem_subject(subj, n_classes=8, permutation_test=False):
    X, _, y, _ = ld.load_data(subj, n_train=500, n_test=0, n_classes=n_classes, data="epochs")
    if (permutation_test):
        shuffle_idx = np.random.choice(500, 500, replace=False)
        y = y[shuffle_idx]
    model = InvertedEncoder(n_classes)
    scores = model.cross_validate(X, y)
    ml.plot_results(np.arange(16), scores, "IEM_cv", subj)
    return scores

def main():
    score_lst = np.zeros(16)
    count = 0
    for subj in ld.meg_subj_lst:
        scores = run_iem_subject(subj)
        score_lst += scores
        count += 1
    
    score_lst /= count
    ml.plot_results(np.arange(16), score_lst, "IEM_cv", "all")



if __name__ == "__main__":
    main()