import numpy as np
import mne
import sklearn
import matplotlib.pyplot as plt

from scipy.io import loadmat

from DoG import init_gmodel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from file_lists import meg_subj_lst, n_subj_trials, ch_picks, new_beh_lst
import load_data as ld


def load_behavior(subj):
    data = loadmat("../IEM-tutorial/Behavior/Sub%d_beh.mat" % new_beh_lst[subj])
    return data['TgtOrs'].squeeze(), data['ResOrs'].squeeze()

def calc_relative_orientation(x):
    if np.abs(x) > 90:
        x = x - (np.sign(x) * 180)
        return x
    return x

def get_diffs(subj):
    tgt, _ = load_behavior(subj)
    tgt = tgt[:n_subj_trials[subj]]
    diffs = [0]
    for i in range(1, len(tgt)):
        diffs.append(calc_relative_orientation(tgt[i - 1] - tgt[i]))
    diffs = np.array(diffs)
    return diffs


def calc_theta_diff(dec, stim):
    error =  dec - stim
    error[np.abs(error) > 90] = error[np.abs(error) > 90] - np.sign(error[np.abs(error) > 90]) * 180
    return error

def calc_theta_diff_all(dec_all, stim_all):
    theta_diffs = [[] for i in range(16)]
    for i in range(16):
        theta_diffs[i].extend(calc_theta_diff(dec_all[i], stim_all[i]))

    return theta_diffs

def calc_mean_error(dec, stim):
    error = calc_theta_diff(dec, stim)
    error = np.deg2rad(error)
    R = (1 / len(dec)) * np.sum(np.exp(1j * error))
    return np.angle(R)

def calc_mean_error_all(dec_all, stim_all):
    errors = []
    for i in range(16):
        errors.append(calc_mean_error(dec_all[i], stim_all[i]))
        
    return np.array(errors)

def calc_sliding_mean_error(dec, stim, diffs):
    error_list = []
    for i in np.arange(-82, 82.1, 1.0):
        dec_window = dec[np.abs(diffs - i) < 8]
        stim_window = stim[np.abs(diffs - i) < 8]

        error = calc_mean_error(dec_window, stim_window)
        error_list.append(error)

    return np.array(error_list)



def calc_sliding_mean_error_all(dec_all, stim_all, diffs):
    error_list = [[] for i in range(16)]
    for i in range(16):
        error_list[i].extend(calc_sliding_mean_error(dec_all[i], stim_all[i], diffs))

    return np.array(error_list)


def shuffle(X, y):
    shuffle_idx = np.random.choice(len(y), len(y), replace=False)
    return X[shuffle_idx], y[shuffle_idx]

class CircularRegression(object):
    def __init__(self, n_timesteps=16):
        self.n_timesteps = n_timesteps


    def cross_validate(self, X, y, diffs):
        # TODO: cross validation
        kfold = KFold(n_splits=5, shuffle=True)
        y = y.flatten().astype(int)
        

        test_angles = np.zeros((self.n_timesteps, len(y)))
        pred_angles = np.zeros((self.n_timesteps, len(y)))
        diff_list = []

        k_num=0

        for train, test in kfold.split(X, y):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            diff_list.extend(diffs[test])

            y_train = 2 * np.pi * y_train / 180
            y_train= np.stack((np.sin(y_train), np.cos(y_train))).T


            for i in range(self.n_timesteps):
                regr = LinearRegression().fit(X_train[:, :, i], y_train)
                out = regr.predict(X_test[:, :, i])
            
                ang_hat = np.arctan(out[:, 0]/out[:, 1])

                inds = np.where(out[:, 1] < 0)[0]
                ang_hat[inds] = ang_hat[inds] + np.pi
                ang_hat = np.mod(ang_hat, 2*np.pi) * 180 / np.pi / 2

                test_angles[i,len(y_test) * k_num: len(y_test) * (k_num + 1)] = y_test
                pred_angles[i,len(y_test) * k_num: len(y_test) * (k_num + 1)] = ang_hat

            k_num += 1
            
        return pred_angles, test_angles, np.array(diff_list)




def data_loader(time_shift=0, mode="sklearn", sample_rate=40, picked_chans=ch_picks):
    saved_data = {}
    def load_data(subj, n_classes=-1):
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

def run_ptest(loader, n_p_tests=100, n_exp_tests=10):
    theta_diffs = [[] for i in range(16)]
    all_diffs = []
    window = np.arange(-82, 82.1, 1.0)
    error_list = np.zeros((16, window.shape[0]))
    for i in range(n_exp_tests):
        print("Experimental Test: %i" % i)
        for subj in meg_subj_lst:
            X, y = loader(subj)
            X, y = shuffle(X, y)
            model = CircularRegression(n_timesteps=16)
            diffs = get_diffs(subj)
            dec, stim, diff_list = model.cross_validate(X, y, diffs)
            all_diffs.extend(diff_list)
            error_list += calc_sliding_mean_error_all(dec, stim, diff_list)
            theta_diff_trial = calc_theta_diff_all(dec, stim)
            for j in range(16):
                theta_diffs[j].extend(theta_diff_trial[j])



    all_diffs = np.array(all_diffs)
    error_list = error_list / (len(meg_subj_lst) * n_exp_tests)

    for i in range(16):
        gmodel = init_gmodel()
        t_error = np.array(theta_diffs[i])


        t_diffs = all_diffs[np.abs(t_error) < 30]
        t_error = t_error[np.abs(t_error) < 30]

        t_error = t_error[np.abs(t_diffs) < 60]
        t_diffs = t_diffs[np.abs(t_diffs) < 60]

        result = gmodel.fit(t_error, x=t_diffs, b=0.03)
        sorted_indices = np.argsort(t_diffs)
        sorted_diffs = t_diffs[sorted_indices]
        sorted_theta_diffs = result.best_fit[sorted_indices]

        plt.figure(figsize=(9, 6))
        plt.scatter(t_diffs, t_error, alpha=0.25)
        plt.plot(sorted_diffs, sorted_theta_diffs, label="Best Fit", color="r", linewidth=4)
        plt.legend()
        plt.savefig("../Figures/circ_reg/DoG%i.png" % i)
        plt.clf()

        plt.figure(figsize=(9, 6))
        plt.plot(window, error_list[i])
        plt.xlim((-90, 90))
        plt.ylim((-10, 10))
        plt.savefig("../Figures/circ_reg/sliding%i.png" % i)
    




def main():
    loader = data_loader()
    run_ptest(loader, n_exp_tests=1)
    



if __name__ == "__main__":
    main()