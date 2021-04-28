import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from file_lists import new_beh_lst, meg_subj_lst, ch_picks, n_subj_trials
import load_data as ld
from DoG import init_gmodel

saved_data = {}

def load_data(subj, n_classes=9, time_shift=0):
    if subj in saved_data:
        mat_dict = saved_data[subj]
        return mat_dict["trn"], mat_dict["trng"], mat_dict["ts"]
    X, _, y, _ = ld.load_data(subj, n_train=n_subj_trials[subj], n_test=0, n_classes=n_classes, data="epochs", 
                                shuffle=False, ch_picks=ch_picks, time_shift=time_shift)
    y = y * (180 / n_classes)
    ts = np.linspace(0, 0.4, 16)
    mat_dict = {
        "trng" : y,
        "trn" : X,
        "ts" : ts
    }
    saved_data[subj] = mat_dict
    return X, y, ts


def load_behavior(subj):
    data = loadmat("../IEM-tutorial/Behavior/Sub%d_beh.mat" % new_beh_lst[subj])
    return data['TgtOrs'].squeeze(), data['ResOrs'].squeeze()

def calc_relative_orientation(x):
    if np.abs(x) > 90:
        x = x - (np.sign(x) * 180)
        return x
    return x


def load_percept_data(subj, n_classes=9, time_shift=0):
    _, trng = load_behavior(subj)
    trng = trng.squeeze()
    trn, _, ts = load_data(subj, n_classes=n_classes, time_shift=time_shift)
    
    trng = trng[:500]
    trn = trn[~np.isnan(trng)]
    trng = trng[~np.isnan(trng)]
    trng[np.abs(trng) > 90] = trng[np.abs(trng) > 90] - 180 * np.sign(trng[np.abs(trng) > 90])
    offset = 180 / (n_classes * 2)
    trng = (trng + 90 + offset) % 180
    trng = np.floor(trng / (180 / n_classes))
    trng = np.minimum(trng, n_classes-1)
    trng = trng * (180 / n_classes)


    return trn, trng, ts

class InvertedEncoder(object):
    def __init__(self, n_ori_chans):
        self.n_ori_chans = n_ori_chans
        self.chan_center = np.linspace(180 / n_ori_chans, 180, n_ori_chans)
        self.xx = np.linspace(1, 180, 180)


    def make_basis_set(self):
        make_basis_function = lambda xx, mu: np.power(np.cos(np.deg2rad(xx - mu)), self.n_ori_chans - (self.n_ori_chans % 2))
        basis_set = np.zeros((180, self.n_ori_chans))
        for cc in range(self.n_ori_chans):
            basis_set[:, cc] = make_basis_function(self.xx, self.chan_center[cc])
        return basis_set

    def make_stimulus_mask(self, trng):
        stim_mask = np.zeros((len(trng), len(self.xx)))
        for tt in range(stim_mask.shape[0]):
            stim_mask[tt, trng[tt]-1] = 1
        return stim_mask

    def make_trn_repnum(self, trng):
        trn_ou = np.unique(trng)
        trn_repnum = np.zeros(len(trng))
        trn_repnum[:] = np.nan
        n_trials_per_orientation = np.zeros(len(trn_ou))

        for ii in range(len(trn_ou)):
            n_trials_per_orientation[ii] = len(trn_repnum[trng == trn_ou[ii]])
            trn_repnum[trng == trn_ou[ii]] = np.arange(n_trials_per_orientation[ii])

        trn_repnum[trn_repnum >= np.min(n_trials_per_orientation)] = np.nan
        return trn_repnum

    

    def cross_validate(self, trnX_cv, trn_cv, trng_cv, trn_repnum):
        trn_cv_coeffs = np.zeros((len(trng_cv), 2 * trn_cv.shape[1], trn_cv.shape[2]))
        trn_cv_coeffs[:, :trn_cv.shape[1], :] = np.real(trn_cv)
        trn_cv_coeffs[:, trn_cv.shape[1]:, :] = np.imag(trn_cv)
    

        chan_resp_cv_coeffs = np.zeros((trn_cv_coeffs.shape[0], len(self.chan_center), trn_cv_coeffs.shape[2]))

        n_reps = int(np.max(trn_repnum))

        for ii in range(n_reps):
            trnidx = trn_repnum != ii
            tstidx = trn_repnum == ii

            thistrn = trn_cv_coeffs[trnidx, :, :]
            thistst = trn_cv_coeffs[tstidx, :, :]

            for tt in range(thistrn.shape[2]):
                thistrn_tpt = thistrn[:, :, tt]
                thistst_tpt = thistst[:, :, tt]

                w_coeffs = np.linalg.lstsq(trnX_cv[trnidx, :], thistrn_tpt)[0]

                chan_resp_cv_coeffs[tstidx, :, tt] = np.linalg.lstsq(w_coeffs.T, thistst_tpt.T)[0].T

        return chan_resp_cv_coeffs

    def run_subject(self, subj, permutation_test=False, shuffle_data=False, plot=False, percept_data=False, shuffle_idx=[], time_shift=0):
        trn, trng, ts = load_percept_data(subj, self.n_ori_chans, time_shift) if percept_data else load_data(subj, self.n_ori_chans, time_shift)
        basis_set = self.make_basis_set()


        if shuffle_data:
            if len(shuffle_idx) == 0:
                shuffle_idx = np.random.choice(len(trng), len(trng), replace=False)
            trn = trn[shuffle_idx]
            trng = trng[shuffle_idx]    

        trng = trng % 180
        trng[trng == 0] = 180
        trng = trng.astype(int)
        trn_repnum = self.make_trn_repnum(trng)
        trng_cv = trng[~np.isnan(trn_repnum)]
        trn_cv = trn[~np.isnan(trn_repnum)]

        if permutation_test:
            np.random.shuffle(trng_cv)

        stim_mask = self.make_stimulus_mask(trng_cv)
        trnX_cv = stim_mask @ basis_set

        trn_repnum = trn_repnum[~np.isnan(trn_repnum)]

        coeffs = self.cross_validate(trnX_cv, trn_cv, trng_cv, trn_repnum)

        targ_ori = int(np.round(len(self.chan_center) / 2))
        coeffs_shift = np.zeros(coeffs.shape)
        for ii in range(self.n_ori_chans):
            idx = trng_cv == self.chan_center[ii]
            coeffs_shift[idx, :, :] = np.roll(coeffs[idx, :, :], targ_ori - ii, axis=1)

        if plot:
            tmean = coeffs_shift.mean(axis=2)

            plt.figure(figsize=(4, 8))
            plt.plot(tmean.mean(axis=0))
            plt.ylim(0, 0.3)
            plt.savefig("../Figures/IEM/python_files/%s_response.png" % subj)
            plt.clf()

        return coeffs_shift, trng_cv, targ_ori

    def get_diffs(self, subj):
        tgt, _ = load_behavior(subj)
        tgt = tgt[:n_subj_trials[subj]]
        diffs = [0]
        for i in range(1, len(tgt)):
            diffs.append(calc_relative_orientation(tgt[i - 1] - tgt[i]))
        diffs = np.array(diffs)
        return diffs

    def run_sd_subject(self, subj, n_bins=30, permutation_test=False):
        _, trng, _ = load_data(subj)
        trn_repnum = self.make_trn_repnum(trng)
        diffs = self.get_diffs(subj)
        diffs = diffs[~np.isnan(trn_repnum)]
        if permutation_test:
            np.random.shuffle(diffs)
        shuffle_idx = np.random.choice(len(trng), len(trng), replace=False)
        coeffs_shift, trng_cv, targ_ori = self.run_subject(subj, shuffle_data=True, shuffle_idx=shuffle_idx)
        bins = np.zeros((n_bins, self.n_ori_chans, coeffs_shift.shape[2]))
        bin_sizes = np.zeros(n_bins)
        diffs = np.minimum(np.floor((diffs + 90) / (180 / n_bins)), n_bins - 1).astype(int)
        for i in range(len(trng_cv)):
            bins[diffs[i]] += coeffs_shift[i]
            bin_sizes[diffs[i]] += 1
        
        return bins, bin_sizes

    def run_sd_dog(self, subj, permutation_test=False):
        _, trng, _ = load_data(subj)
        trn_repnum = self.make_trn_repnum(trng)
        diffs = self.get_diffs(subj)
        diffs = diffs[~np.isnan(trn_repnum)]
        if permutation_test:
            np.random.shuffle(diffs)
        shuffle_idx = np.random.choice(len(trng), len(trng), replace=False)
        coeffs_shift, _, _ = self.run_subject(subj, shuffle_data=True, shuffle_idx=shuffle_idx)
        
        return coeffs_shift, diffs
            


def n_correct(tmean, targ_ori, n_trials):
    n = 0
    for i in range(n_trials):
        if np.argmax(tmean[i]) == targ_ori:
            n += 1
    return n
    
def n_correct_tsteps(coeffs, targ_ori, n_trials):
    n_timesteps = coeffs.shape[2]
    n = np.zeros(n_timesteps)
    for i in range(n_trials):
        for j in range(n_timesteps):
            if np.argmax(coeffs[i, :, j]) == targ_ori:
                n[j] += 1
    return n

#subjlist = ["AK", "DI", "HHy", "HN", "JL", "KA", "MF", "NN", "SoM", "TE", "VA", "YMi"]

def make_pd_bar(exp_accs, perm_accs):
    data_lst = []
    data_lst.extend(exp_accs)
    data_lst.extend(perm_accs)

    str_lst = []
    str_lst.extend(["experimental" for _ in exp_accs])
    str_lst.extend(["permutation" for _ in perm_accs])

    d = {'accuracy': data_lst, 'Trial': str_lst}
    df = pd.DataFrame(data=d)
    return df

def run_all_subjects(n_ori_chans, n_p_tests=100, n_exp_tests=20, n_timesteps=16, percept_data=False, time_shift=0):
    IEM = InvertedEncoder(n_ori_chans)
    avg_response = np.zeros(n_ori_chans)
    total_trials = 0
    targ_ori = -1

    print("Running Experimental Test")
    exp_list_x = []
    exp_list_y = []
    exp_accuracies = np.zeros(n_exp_tests)
    exp_t_accs_x = []
    exp_t_accs_y = []
    exp_ch_responses = np.zeros((n_exp_tests, n_ori_chans))
    exp_results = np.zeros((n_exp_tests, n_timesteps))
    exp_timestep_ch_response = np.zeros((n_ori_chans, n_timesteps))
    exp_accuracies_4_8 = np.zeros(n_exp_tests)
    for i in range(n_exp_tests):
        print("Trial %d" % (i + 1))
        test_trials = 0
        trial_response = np.zeros(n_ori_chans)
        trial_accuracy = np.zeros(n_timesteps)
        for subj in meg_subj_lst:
            coeffs, trng_cv, targ_ori = IEM.run_subject(subj, shuffle_data=True, 
                                                        percept_data=percept_data, time_shift=time_shift)    
            tmean = coeffs.mean(axis=2)
            exp_timestep_ch_response += coeffs.mean(axis=0) * len(trng_cv)
            exp_accuracies[i] += n_correct(tmean, targ_ori, len(trng_cv))
            exp_accuracies_4_8[i] += n_correct(coeffs[:, :, 4:9].mean(axis=2), targ_ori, len(trng_cv))
            trial_accuracy += n_correct_tsteps(coeffs, targ_ori, len(trng_cv))
            trial_response += np.sum(tmean, 0)
            total_trials += len(trng_cv)
            test_trials += len(trng_cv)
        exp_accuracies[i] /= test_trials
        exp_accuracies_4_8[i] /= test_trials

        trial_accuracy = trial_accuracy / test_trials
        exp_t_accs_x.extend(np.linspace(-0.4, -0.025, n_timesteps))
        exp_t_accs_y.extend(trial_accuracy)

        avg_response += trial_response
        trial_response = trial_response / test_trials
        exp_ch_responses[i] += trial_response
        exp_list_x.extend(np.arange(n_ori_chans))
        exp_list_y.extend(trial_response)
        exp_results[i] += trial_accuracy
        print("Test Accuracy: {:.3f}".format(exp_accuracies[i]))

    avg_response = avg_response / total_trials
    exp_timestep_ch_response /= total_trials
    plt.figure(figsize=(8, 8))
    plt.imshow(exp_timestep_ch_response, aspect="equal")
    plt.colorbar()
    plt.xticks(ticks=np.arange(-0.5, n_timesteps+0.5, 1), labels=np.arange(-400, -25, 25).astype(int))
    plt.yticks(ticks=np.arange(-0.5, n_ori_chans+0.5, 1), labels=np.arange(0, 180, 180 / n_ori_chans).astype(int))
    plt.xlabel("Timestep (millisencods)")
    plt.ylabel("Channel")

    plt.title("Channel Response at each timestep")
    plt.savefig("../Figures/IEM/python_files/timestep_ch_response.png")
    plt.clf()

    print("Running Permutation Tests")
    permutation_response = np.zeros(n_ori_chans)
    extreme_accs = np.zeros(n_exp_tests)
    extreme_accs_4_8 = np.zeros(n_exp_tests)
    perm_t_accs_x = []
    perm_t_accs_y = []
    perm_accuracy = 0
    perm_list_x = []
    perm_list_y = []
    perm_accuracies = []
    extreme_ch_resps = np.zeros(n_ori_chans)
    perm_results = np.zeros((n_p_tests, n_timesteps))
    for i in range(n_p_tests):
        print("Permutation %d" % (i + 1))
        total_response = np.zeros(n_ori_chans)
        total_trials = 0
        temp_perm_accuracy = 0
        perm_trial_acc = np.zeros(n_timesteps)

        for subj in meg_subj_lst:
            coeffs, trng_cv, targ_ori = IEM.run_subject(subj, shuffle_data=True, permutation_test=True, 
                                                        percept_data=percept_data, time_shift=time_shift)
            tmean = coeffs.mean(axis=2)
            temp_perm_accuracy += n_correct(tmean, targ_ori, len(trng_cv))
            perm_trial_acc += n_correct_tsteps(coeffs, targ_ori, len(trng_cv))
            total_response += np.sum(tmean, 0)
            total_trials += len(trng_cv)
        temp_perm_accuracy /= total_trials
        print("accuracy: {:.3f}".format(temp_perm_accuracy))
        perm_accuracy += temp_perm_accuracy
        for j in range(n_exp_tests):
            if temp_perm_accuracy >= exp_accuracies[j]:
                extreme_accs[j] += 1
            if temp_perm_accuracy >= exp_accuracies_4_8[j]:
                extreme_accs_4_8[j] += 1
        perm_accuracies.append(temp_perm_accuracy)

        perm_trial_acc = perm_trial_acc / total_trials
        perm_t_accs_x.extend(np.linspace(-0.4, -0.025, n_timesteps))
        perm_t_accs_y.extend(perm_trial_acc)

        perm_avg_response = total_response / total_trials
        for j in range(n_ori_chans):
            for k in range(n_exp_tests):
                if perm_avg_response[j] >= exp_ch_responses[k][j]:
                    extreme_ch_resps[j] += 1

        perm_list_x.extend(np.arange(n_ori_chans))
        perm_list_y.extend(perm_avg_response)
        permutation_response += perm_avg_response
        perm_results[i] += perm_trial_acc

    ch_resp_p_values = extreme_ch_resps / (n_exp_tests * n_p_tests)
    acc_df = make_pd_bar(exp_accuracies, perm_accuracies)
    permutation_response = permutation_response / n_p_tests
    perm_accuracy /= n_p_tests
    print("accuracy: {:.3f}".format(perm_accuracy))

    plt.figure(figsize=(8, 8))
    sns.barplot(x='Trial', y='accuracy', data=acc_df, ci="sd")
    plt.title("Mean Accuracy, p-value: {:.3f}".format(extreme_accs.mean() / n_p_tests))
    plt.savefig("../Figures/IEM/python_files/accuracy.png")
    plt.clf()

    acc_df_4_8 = make_pd_bar(exp_accuracies_4_8, perm_accuracies)
    plt.figure(figsize=(8, 8))
    sns.barplot(x='Trial', y='accuracy', data=acc_df_4_8, ci="sd")
    plt.title("Accuracy from 100-200 ms, p-value: {:.3f}".format(extreme_accs_4_8.mean() / n_p_tests))
    plt.savefig("../Figures/IEM/python_files/accuracy_4_8.png")
    plt.clf()

    plt.figure(figsize=(8,8))
    plt.hist(perm_accuracies, bins=np.linspace(0.09, 0.14, 20), label="permutation")
    plt.axvline(exp_accuracies.mean(), label="Mean Accuracy", linewidth=2.5, color='r')
    plt.title("Accuracy vs. Null Distribution")
    plt.legend()
    plt.xlabel("Accuracy")
    plt.ylabel("Occurences")
    plt.savefig("../Figures/IEM/python_files/null_comparison.png")
    plt.clf()

    plt.figure(figsize=(4, 8))
    sns.lineplot(exp_list_x, exp_list_y, label="Experimental Response", ci="sd")
    sns.lineplot(perm_list_x, perm_list_y, label="Permutation", ci="sd")
    
    plt.ylim(0.1, 0.35)
    for i in range(n_ori_chans):
        plt.text(i-0.5, avg_response[i]+0.03, ch_resp_p_values[i], color="red")
    plt.legend()
    plt.xticks(ticks=np.arange(n_ori_chans), labels=np.arange(0, 180, 180 / n_ori_chans).astype(int))
    plt.title("Mean Channel Response of Experiment vs. Permutation groups")
    plt.xlabel("Orientation Channel (Degrees)")
    plt.ylabel("Channel Response")
    plt.savefig("../Figures/IEM/python_files/perm_response.png")
    plt.clf()


    timestep_accuracy_p_values = np.zeros(n_timesteps)
    for i in range(n_exp_tests):
        for j in range(n_timesteps):
            timestep_accuracy_p_values[j] += len(perm_results[perm_results[:, j] > exp_results[i, j]])

    timestep_accuracy_p_values /= (n_exp_tests * n_p_tests)
    timesteps = np.linspace(-0.4, -0.025, n_timesteps) 
    timestep_width = 0.4 / n_timesteps
    sig_ranges = [[]]
    for i in range(n_timesteps):
        if timestep_accuracy_p_values[i] <= 0.05:
            if len(sig_ranges[-1]) == 0:
                sig_ranges[-1] = [timesteps[i] - (timestep_width / 2), timesteps[i] + (timestep_width / 2)]
            else:
                sig_ranges[-1][1] = timesteps[i] + (timestep_width / 2)
        else:
            if len(sig_ranges[-1]) > 0:
                sig_ranges += [[]]  

    plt.figure(figsize=(8, 8))
    sns.lineplot(exp_t_accs_x, exp_t_accs_y, label="Experimental Accuracy", ci="sd")
    sns.lineplot(perm_t_accs_x, perm_t_accs_y, label="Permutation Accuracy", ci="sd")
    for rng in sig_ranges:
        if len(rng) > 0:
            plt.axvspan(max(rng[0], -0.4), rng[1], color="lightgreen", alpha=0.25)
    
    plt.legend()
    plt.ylim(0.05, 0.15)
    plt.title("Accuracy at each timestep")
    plt.savefig("../Figures/IEM/python_files/timestep_accuracy.png")
    plt.clf()

def calc_sd_bias(coeffs, timestep):
    weights = np.array([0, -0.25, -0.5, -1, 0, 1, 0.5, 0.25, 0]) * 20

    return np.dot(np.minimum(np.maximum(coeffs[:, :, timestep], 0), 1), weights)

def run_DoG_trial(ch_bias, diffs, bootstrap_size=10000, permutation_test=False):
    bootstrap_idx = np.random.choice(len(diffs), size=bootstrap_size, replace=True)
    ch_bias_bootstrap = ch_bias[bootstrap_idx]
    diffs_bootstrap = diffs[bootstrap_idx]

    diffs_bootstrap = diffs_bootstrap[np.abs(ch_bias_bootstrap) <= 40]
    ch_bias_bootstrap = ch_bias_bootstrap[np.abs(ch_bias_bootstrap) <= 40]

    ch_bias_bootstrap = ch_bias_bootstrap[np.abs(diffs_bootstrap) <= 60]
    diffs_bootstrap = diffs_bootstrap[np.abs(diffs_bootstrap) <= 60]

    if permutation_test:
        np.random.shuffle(diffs_bootstrap)
    gmodel = init_gmodel()
    result = gmodel.fit(ch_bias_bootstrap, x=diffs_bootstrap, b=0.03)
    return result.best_fit, diffs_bootstrap, result.params["a"]

def iem_sd_dog(n_ori_chans, n_exp_tests=2, n_bootstraps=1000, n_permutations=10000, n_timesteps=16):
    model = InvertedEncoder(n_ori_chans)
    ch_bias_list = [[] for _ in range(16)]
    diffs_list = []


    for i in range(n_exp_tests):
        print("Experimental Test: %i" % i)

        for subj in meg_subj_lst:
            coeffs, diffs = model.run_sd_dog(subj)
            diffs_list.extend(diffs)

            for j in range(n_timesteps):
                ch_bias_list[j].extend(calc_sd_bias(coeffs, j))


    diffs_list = np.array(diffs_list)
    ch_bias = np.array(ch_bias_list)
    print(ch_bias[0])    

    
    a_list = np.zeros((n_timesteps, n_bootstraps))
    p_values = np.zeros(n_timesteps)
    for j in range(n_timesteps):
        print("timestep: %d" % j)
        dog_fit_list = []
        diffs_bootstrap_list = []

        dog_fit_p_list = []
        diffs_bootstrap_p_list = []

        for i in range(n_bootstraps):
            dog_fit, diffs_bootstrap, a_bootstrap = run_DoG_trial(ch_bias[j], diffs)
            a_list[j, i] = a_bootstrap
            if i % 10 == 0:
                dog_fit_list.extend(dog_fit)
                diffs_bootstrap_list.extend(diffs_bootstrap)
            
        for i in range(n_permutations):
            dog_fit, diffs_bootstrap, a_bootstrap = run_DoG_trial(ch_bias[j], diffs, permutation_test=True)
            p_values[j] += len(a_list[j][a_list[j] <= a_bootstrap])
            if i % 100 == 0:
                dog_fit_p_list.extend(dog_fit)
                diffs_bootstrap_p_list.extend(diffs_bootstrap)

        p_values[j] /= (n_permutations * n_bootstraps)

        plt.figure(figsize=(9, 6))
        plt.scatter(diffs_list, ch_bias[j], alpha=0.25)
        sns.lineplot(diffs_bootstrap_list, dog_fit_list, color="r", label="Bootstrapped DoG", linewidth=3.5)
        sns.lineplot(diffs_bootstrap_p_list, dog_fit_p_list, color="g", label="Permutation DoG", linewidth=3.5)
        plt.legend()
        plt.xlabel("Relative orientation of previous trial")
        plt.ylabel("Channel response bias on current trial")
        plt.xlim((-60, 60))
        plt.ylim((-40, 40))
        plt.title("a={:.3f}        P={:.3f}".format(a_list[j].mean(), p_values[j]))
        plt.savefig("../Figures/final_results/bias/bias_dog_t%d.png" % j)
        plt.clf()

    print(a_list.mean(1))
    print(a_list.max(0))
    plt.figure(figsize=(9, 6))
    xs = np.linspace(0, 0.375, 16)
    ys = a_list.mean(axis=1)
    plt.bar(xs, ys, width=0.025)
    for i in range(9):
        if p_values[i] <= 0.05:
            plt.text(xs[i], ys[i]+0.01, "*")
    plt.savefig("../Figures/final_results/bias/bias_all.png")
            
    return


def iem_sd_all(n_ori_chans, n_bins=15, percept_data=False, n_p_tests=100, n_exp_tests=25, n_timesteps=16):
    IEM = InvertedEncoder(n_ori_chans)
    avg_response = np.zeros((n_ori_chans, n_timesteps))
   
    bin_accuracies = np.zeros((n_exp_tests, n_bins, n_ori_chans, n_timesteps))
    for i in range(n_exp_tests):
        print("Experimental Test: %i" % i)
        bin_sizes = np.zeros(n_bins)
        bins = np.zeros((n_bins, n_ori_chans, n_timesteps))
        for subj in meg_subj_lst:
            temp_bins, temp_bin_sizes = IEM.run_sd_subject(subj, n_bins=n_bins)
            bins += temp_bins
            bin_sizes += temp_bin_sizes
        
        for j in range(n_timesteps):
            bin_accuracies[i, :, :, j] = bins[:, :, j] / bin_sizes[:, None]
            avg_response[:, j] += (np.sum(bins[:, :, j], axis=0) / np.sum(bin_sizes))

    avg_response /= (n_exp_tests)
    perm_bin_accuracies = np.zeros((n_p_tests, n_bins, n_ori_chans, n_timesteps))
    for i in range(n_p_tests):
        print("Permutation Test: %i" %i)
        bin_sizes = np.zeros(n_bins)
        bins = np.zeros((n_bins, n_ori_chans, n_timesteps))
        for subj in meg_subj_lst:
            temp_bins, temp_bin_sizes = IEM.run_sd_subject(subj, n_bins=n_bins, permutation_test=True)
            bins += temp_bins
            bin_sizes += temp_bin_sizes
        
        for j in range(n_timesteps):
            perm_bin_accuracies[i, :, :, j] = bins[:, :, j] / bin_sizes[:, None]

    perm_avg_bin_accuracies = np.zeros(perm_bin_accuracies.shape)
    avg_bin_accuracies = np.zeros(bin_accuracies.shape)
    for i in range(n_p_tests):
        for j in range(n_timesteps):
            perm_avg_bin_accuracies[i, :, :, j] += perm_bin_accuracies[i, :, :, j] - avg_response[:, j]
        
    for i in range(n_exp_tests):
        for j in range(n_timesteps):
            avg_bin_accuracies[i, :, :, j] += bin_accuracies[i, :, :, j] - avg_response[:, j]

    ptest_comp = np.zeros((n_bins, n_ori_chans, n_timesteps))
    for i in range(n_p_tests):
        for j in range(n_exp_tests):
            gr_eq_vals = np.abs(perm_avg_bin_accuracies[i, :, :, :]) >= np.abs(avg_bin_accuracies[j, :, :, :])
            ptest_comp[gr_eq_vals] += 1.0
    ptest_comp /= (n_p_tests * n_exp_tests)
        
    for j in range(n_timesteps):
        figure = plt.figure(figsize=(8,15))

        

        ax0 = figure.add_subplot(3, 1, 1)
        im0 = ax0.imshow(bin_accuracies[:, :, :, j].mean(axis=0).T, aspect="equal", vmin=0.15, vmax=0.35)
        ax0.set_xlabel("Relative Previous Orientation")
        figure.colorbar(im0, ax=ax0)
        plt.xticks(ticks=np.arange(-0.5, n_bins+0.5, 1), labels=np.linspace(-90, 90, n_bins+1).astype(int))
        plt.yticks(ticks=np.arange(-0.5, n_ori_chans+0.5, 1), labels=np.arange(0, 180, 180 / n_ori_chans).astype(int))
        plt.ylabel("Channel response")
        plt.title("Channel response binned by previous orientation, t=%d" % ((j + 16) * 25))

        ax1 = figure.add_subplot(3, 1, 2)
        im1 = ax1.imshow(perm_bin_accuracies[:, :, :, j].mean(axis=0).T, aspect="equal", vmin=0.15, vmax=0.35)
        ax1.set_xlabel("Relative Previous Orientation")
        figure.colorbar(im1, ax=ax1)

        plt.xticks(ticks=np.arange(-0.5, n_bins+0.5, 1), labels=np.linspace(-90, 90, n_bins+1).astype(int))
        plt.yticks(ticks=np.arange(-0.5, n_ori_chans+0.5, 1), labels=np.arange(0, 180, 180 / n_ori_chans).astype(int))
        plt.ylabel("Channel response")
        plt.title("Permutation")

        ax2 = figure.add_subplot(3, 1, 3)
        im2 = ax2.imshow(ptest_comp[:, :, j].T, aspect="equal", vmin=0, vmax=1)
        ax2.set_xlabel("Relative Previous Orientation")
        figure.colorbar(im2, ax=ax2)

        plt.xticks(ticks=np.arange(-0.5, n_bins+0.5, 1), labels=np.linspace(-90, 90, n_bins+1).astype(int))
        plt.yticks(ticks=np.arange(-0.5, n_ori_chans+0.5, 1), labels=np.arange(0, 180, 180 / n_ori_chans).astype(int))
        plt.ylabel("Channel response")
        plt.title("P Value")
        
        
        
        plt.savefig("../Figures/IEM/python_files/sd_all_t%d.png" %  ((j + 16) * 25))
        plt.clf()
    return

def iem_sd_all_v2(n_ori_chans, n_bins=15, percept_data=False, n_p_tests=100, n_exp_tests=25, n_timesteps=16):
    IEM = InvertedEncoder(n_ori_chans)
    avg_response = np.zeros((n_ori_chans, n_timesteps))

    coeffs_list = []
    diffs_list = []
   
    bin_accuracies = np.zeros((n_exp_tests, n_bins, n_ori_chans, n_timesteps))
    for i in range(n_exp_tests):
        print("Experimental Test: %i" % i)
        for subj in meg_subj_lst:
            coeffs, diffs = IEM.run_sd_dog(subj)
            coeffs_list.extend(coeffs)
            diffs_list.extend(diffs)
        
    bins = np.zeros((n_bins, n_ori_chans, n_timesteps))
    
    coeffs_list = np.array(coeffs_list)
    diffs_list = np.array(diffs_list).astype(int)

    for i in range(n_bins):
        bins[i] = np.sum(coeffs_list[diffs_list == i], axis=0)
        bins[i] /= len(diffs_list[diffs_list == i])

    avg_response = coeffs_list.mean(axis=0)

    perm_bin_sizes = np.zeros(n_bins)
    perm_bins = np.zeros((n_bins, n_ori_chans, n_timesteps))
    perm_diffs_list = diffs_list[:]
    for i in range(n_p_tests):
        for j in range(n_bins):
            np.random.shuffle(perm_diffs_list)
            perm_bins[j] += np.sum(coeffs_list[perm_diffs_list == j], axis=0)
            perm_bin_sizes[j] += len(perm_diffs_list[perm_diffs_list==j])

    for i in range(n_bins):
        perm_bins[i] /= perm_bin_sizes[i]

    



    perm_bin_accuracies = np.zeros((n_p_tests, n_bins, n_ori_chans, n_timesteps))
    for i in range(n_p_tests):
        print("Permutation Test: %i" %i)
        bin_sizes = np.zeros(n_bins)
        bins = np.zeros((n_bins, n_ori_chans, n_timesteps))
        for subj in meg_subj_lst:
            temp_bins, temp_bin_sizes = IEM.run_sd_subject(subj, n_bins=n_bins, permutation_test=True)
            bins += temp_bins
            bin_sizes += temp_bin_sizes
        
        for j in range(n_timesteps):
            perm_bin_accuracies[i, :, :, j] = bins[:, :, j] / bin_sizes[:, None]

    perm_avg_bin_accuracies = np.zeros(perm_bin_accuracies.shape)
    avg_bin_accuracies = np.zeros(bin_accuracies.shape)
    for i in range(n_p_tests):
        for j in range(n_timesteps):
            perm_avg_bin_accuracies[i, :, :, j] += perm_bin_accuracies[i, :, :, j] - avg_response[:, j]
        
    for i in range(n_exp_tests):
        for j in range(n_timesteps):
            avg_bin_accuracies[i, :, :, j] += bin_accuracies[i, :, :, j] - avg_response[:, j]

    ptest_comp = np.zeros((n_bins, n_ori_chans, n_timesteps))
    for i in range(n_p_tests):
        for j in range(n_exp_tests):
            gr_eq_vals = np.abs(perm_avg_bin_accuracies[i, :, :, :]) >= np.abs(avg_bin_accuracies[j, :, :, :])
            ptest_comp[gr_eq_vals] += 1.0
    ptest_comp /= (n_p_tests * n_exp_tests)
        
    for j in range(n_timesteps):
        figure = plt.figure(figsize=(8,15))

        

        ax0 = figure.add_subplot(3, 1, 1)
        im0 = ax0.imshow(bin_accuracies[:, :, :, j].mean(axis=0).T, aspect="equal", vmin=0.15, vmax=0.35)
        ax0.set_xlabel("Relative Previous Orientation")
        figure.colorbar(im0, ax=ax0)
        plt.xticks(ticks=np.arange(-0.5, n_bins+0.5, 1), labels=np.linspace(-90, 90, n_bins+1).astype(int))
        plt.yticks(ticks=np.arange(-0.5, n_ori_chans+0.5, 1), labels=np.arange(0, 180, 180 / n_ori_chans).astype(int))
        plt.ylabel("Channel response")
        plt.title("Channel response binned by previous orientation, t=%d" % j)

        ax1 = figure.add_subplot(3, 1, 2)
        im1 = ax1.imshow(perm_bin_accuracies[:, :, :, j].mean(axis=0).T, aspect="equal", vmin=0.15, vmax=0.35)
        ax1.set_xlabel("Relative Previous Orientation")
        figure.colorbar(im1, ax=ax1)

        plt.xticks(ticks=np.arange(-0.5, n_bins+0.5, 1), labels=np.linspace(-90, 90, n_bins+1).astype(int))
        plt.yticks(ticks=np.arange(-0.5, n_ori_chans+0.5, 1), labels=np.arange(0, 180, 180 / n_ori_chans).astype(int))
        plt.ylabel("Channel response")
        plt.title("Permutation")

        ax2 = figure.add_subplot(3, 1, 3)
        im2 = ax2.imshow(ptest_comp[:, :, j].T, aspect="equal", vmin=0, vmax=1)
        ax2.set_xlabel("Relative Previous Orientation")
        figure.colorbar(im2, ax=ax2)

        plt.xticks(ticks=np.arange(-0.5, n_bins+0.5, 1), labels=np.linspace(-90, 90, n_bins+1).astype(int))
        plt.yticks(ticks=np.arange(-0.5, n_ori_chans+0.5, 1), labels=np.arange(0, 180, 180 / n_ori_chans).astype(int))
        plt.ylabel("Channel response")
        plt.title("P Value")
        
        
        
        plt.savefig("../Figures/IEM/python_files/sd_all_t%d.png" % j)
        plt.clf()
    return

def main():
    #run_all_subjects(9, percept_data=False)
    run_all_subjects(9, time_shift=-1)
    #load_behavior("KA")
    #load_percept_data("KA")
    #iem_sd_all(9)
    #iem_sd_dog(9, n_exp_tests=25, n_bootstraps=1000, n_permutations=10000)

if __name__ == "__main__":
    main()
