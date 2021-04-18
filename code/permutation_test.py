import machine_learning as ml
import numpy as np
import matplotlib.pyplot as plt
import load_data as ld
from file_lists import ch_picks, meg_subj_lst, n_subj_trials
import seaborn as sns
import pandas as pd
from scipy.io import loadmat, savemat

def data_loader(time_shift=0, mode="sklearn", sample_rate=40, picked_chans=ch_picks):
    saved_data = {}
    def load_data(subj, n_classes=9):
        if subj in saved_data:
            mat_dict = saved_data[subj]
            return mat_dict["X"], mat_dict["y"]
        X, _, y, _ = ld.load_data(subj, n_train=n_subj_trials[subj], n_test=0, n_classes=n_classes, shuffle=True, data="epochs", 
                                    mode=mode, ch_picks=picked_chans, time_shift=time_shift, sample_rate=sample_rate)
        saved_data[subj] = {
            "X" : X,
            "y" : y
        }
        return X, y
    return load_data

def percept_data_loader(mode="sklearn", time_shift=0, sample_rate=40, picked_chans=ch_picks):
    saved_data = {}
    def load_data(subj, n_classes=9):
        if subj in saved_data:
            mat_dict = saved_data[subj]
            return mat_dict["X"], mat_dict["y"].squeeze()

        y, _ = ld.load_behavior(subj)
        y = y.squeeze()
        y = y[:n_subj_trials[subj]]
        X, _, _, _ = ld.load_data(subj, n_train=n_subj_trials[subj], n_test=0, n_classes=n_classes, shuffle=True, data="epochs", 
                                    mode=mode, ch_picks=picked_chans, time_shift=time_shift, sample_rate=sample_rate)

        X = X[~np.isnan(y)]
        y = y[~np.isnan(y)]
        y[np.abs(y) > 90] = y[np.abs(y) > 90] - 180 * np.sign(y[np.abs(y) > 90])
        offset = 180 / (n_classes * 2)
        y = (y + 90 + offset) % 180
        y = np.floor(y / (180 / n_classes))
        y = np.minimum(y, n_classes-1)
        
        saved_data[subj] = {
            "X": X,
            "y": y
        }
        return X, y
    return load_data

def data_loader_source(time_shift=0):
    saved_data = {}
    def load_source(subj, n_classes=9):
        if subj in saved_data:
            mat_dict = loadmat("../Data/mat/%s.mat" % subj)
            return mat_dict["X"], mat_dict["y"].squeeze()
        X, _, y, _ = ld.load_data(subj, n_train=600, n_test=0, n_classes=n_classes, shuffle=True, data="stc", ch_picks=[], time_shift=time_shift)
        mat_dict = {
            "X" : X,
            "y" : y
        }
        savemat("../Data/mat/%s.mat" % subj, mat_dict)
        saved_data[subj] = subj
        return X, y
    return load_source

def data_loader_cnn(time_shift=0):
    saved_data = {}
    def load_data(subj, n_classes=9):
        if subj in saved_data:
            mat_dict = saved_data[subj]
            return mat_dict["X"], mat_dict["y"].squeeze()
        X, _, y, _ = ld.load_data(subj, n_train=600, n_test=0, n_classes=n_classes, shuffle=True, data="wave", ch_picks=ch_picks, time_shift=time_shift)
        saved_data[subj] = {
            "X" : X,
            "y" : y
        }
        return X, y
    return load_data

def make_pd_bar(exp_accs, perm_accs):
    data_lst = []
    data_lst.extend(exp_accs)
    data_lst.extend(perm_accs)

    str_lst = []
    str_lst.extend(["experimental" for _ in exp_accs])
    str_lst.extend(["permutation" for _ in perm_accs])

    d = {'accuracy': data_lst, 'trial': str_lst}
    df = pd.DataFrame(data=d)
    return df

def run_subject(subj, load_data, n_classes=9, permutation=False, model_type="logistic_sensor"):
    X, y = load_data(subj, n_classes)
    repnum = np.zeros(X.shape[0])
    
    n_trials_per_orientation = np.zeros(n_classes)

    for i, value in np.ndenumerate(np.unique(y)):
        n_trials_per_orientation[i] = len(repnum[y == value])
        repnum[y == value] = np.arange(n_trials_per_orientation[i])

    repnum[repnum >= np.min(n_trials_per_orientation)] = np.nan
    X, y = X[~np.isnan(repnum)], y[~np.isnan(repnum)]


    length = len(y)

    shuffle_idx = np.random.choice(length, length, replace=False)
    X, y = X[shuffle_idx], y[shuffle_idx]

    if permutation:
        np.random.shuffle(y)
    model = []
    if model_type == "logistic_sensor":
        model = ml.LogisticSlidingModel(max_iter=1500, n_classes=n_classes, k=25, C=0.09, l1_ratio=0.95)
    elif model_type == "logistic_source":
        model = ml.LogisticSlidingModel(max_iter=4000, n_classes=n_classes, k=1000, C=0.05, l1_ratio=0.95)
    elif model_type == "svm_sensor":
        model = ml.SVMSlidingModel(k=25, C=0.85)
    elif model_type == "cnn_sensor":
        model = ml.CNNSlidingModel(X.shape[1:], n_classes=n_classes)
    elif model_type == "snn_sensor":
        model = ml.DenseSlidingModel(n_classes=n_classes, n_epochs=5)
    results = model.cross_validate(X, y)
    return results, length

def run_ptest(load_data, n_classes=9, n_p_tests=500, n_exp_tests=100, sample_rate=40, model_type="logistic_sensor", identifier="all"):
    n_timesteps = int(np.floor(0.4 * sample_rate))
    print(n_timesteps)
    exp_results = np.zeros((n_exp_tests, n_timesteps))
    
    for i in range(n_exp_tests):
        print("Experimental test: %d" % i)
        exp_trials = 0
        for subj in meg_subj_lst:
            temp_results, trials = run_subject(subj, load_data, n_classes=n_classes, model_type=model_type)
            exp_results[i] += temp_results * trials
            exp_trials += trials
        exp_results[i] /= exp_trials
    mean_exp_acc = exp_results.mean(1)
    print(mean_exp_acc)

    perm_results = np.zeros((n_p_tests, n_timesteps))
    significant_permutations = 0
    for i in range(n_p_tests):
        print("Permutation test: %d" % i)
        perm_trials = 0
        for subj in meg_subj_lst:
            temp_results, trials = run_subject(subj, load_data, n_classes=n_classes, permutation=True, model_type=model_type)
            perm_results[i] += temp_results * trials
            perm_trials += trials
        perm_results[i] /= perm_trials
        for j in range(n_exp_tests):
            if perm_results[i].mean() > mean_exp_acc[j]:
                significant_permutations += 1
        
    p_value = significant_permutations / (n_exp_tests * n_p_tests)


    
    mean_perm_acc = perm_results.mean(1)

    print("Experimental Accuracy {:.3f}".format(mean_exp_acc.mean()))
    print("Permutation Accuracy {:.3f}".format(mean_perm_acc.mean()))

    acc_df = make_pd_bar(mean_exp_acc, mean_perm_acc)
    plt.figure(figsize=(8, 8))
    sns.barplot(x='trial', y='accuracy', data=acc_df, ci="sd")
    plt.title("Mean Accuracy, p-value: {:.3f}".format(p_value))
    plt.savefig("../Figures/final_results/" + model_type + "/accuracy_" + identifier + ".png")
    plt.clf()

    perm_t_accs_x = []
    perm_t_accs_y = []
    exp_t_accs_x = []
    exp_t_accs_y = []
    for i in range(n_exp_tests):
        exp_t_accs_x.extend(np.linspace(0, 0.375, n_timesteps))
        exp_t_accs_y.extend(exp_results[i])

    for i in range(n_p_tests):
        perm_t_accs_x.extend(np.linspace(0, 0.375, n_timesteps))
        perm_t_accs_y.extend(perm_results[i])

    timestep_accuracy_p_values = np.zeros(n_timesteps)
    for i in range(n_exp_tests):
        for j in range(n_timesteps):
            timestep_accuracy_p_values[j] += len(perm_results[perm_results[:, j] > exp_results[i, j]])

    timestep_accuracy_p_values /= (n_exp_tests * n_p_tests)
    timesteps = np.linspace(0, 0.4 - (0.4 / n_timesteps), n_timesteps)
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

    print(timestep_accuracy_p_values)

    plt.figure(figsize=(8, 8))
    sns.lineplot(exp_t_accs_x, exp_t_accs_y, label="Experimental Accuracy", ci="sd")
    sns.lineplot(perm_t_accs_x, perm_t_accs_y, label="Permutation Accuracy", ci="sd")

    for rng in sig_ranges:
        if len(rng) > 0:
            plt.axvspan(max(rng[0], 0), rng[1], color="red", alpha=0.25)

    plt.legend()
    plt.ylim(0.05, 0.15)
    plt.ylabel("Decoding Accuracy")
    plt.xlabel("Time After Stimulus Onset (ms)")
    plt.title("Accuracy at each timestep")
    plt.savefig("../Figures/final_results/" + model_type + "/timestep_accuracy_" + identifier + ".png")
    plt.clf()

def main():
    load_data = percept_data_loader()
    run_ptest(load_data, n_classes=9, model_type="logistic_sensor", identifier="percept_occipital")
    run_ptest(load_data, n_classes=9, model_type="svm_sensor", identifier="percept_occipital")

    load_data = percept_data_loader(picked_chans=[])
    run_ptest(load_data, n_classes=9, model_type="logistic_sensor", identifier="percept_all")
    run_ptest(load_data, n_classes=9, model_type="svm_sensor", identifier="percept_all")

    load_data = data_loader(time_shift=-1)
    run_ptest(load_data, n_classes=9, model_type="logistic_sensor", identifier="prev_occipital")
    run_ptest(load_data, n_classes=9, model_type="svm_sensor", identifier="prev_occipital")

    load_data = data_loader(time_shift=-1, picked_chans=[])
    run_ptest(load_data, n_classes=9, model_type="logistic_sensor", identifier="prev_all")
    run_ptest(load_data, n_classes=9, model_type="svm_sensor", identifier="prev_all")

if __name__ == "__main__":
    main()