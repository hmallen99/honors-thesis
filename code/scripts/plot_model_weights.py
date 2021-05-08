import numpy as np
import matplotlib.pyplot as plt
import mne
import MEG_analysis as meg
import source_localization as srcl
import load_data as ld
import machine_learning  as ml
from file_lists import aligned_dir, meg_locations, meg_subj_lst, n_subj_trials

saved_epochs_info = {}
saved_data = {}

def load_data():
    for subj in meg_subj_lst:
        X, _, y, _ = ld.load_data(subj, n_train=n_subj_trials[subj], n_test=0, n_classes=9, data="epochs", 
                                            shuffle=True, ch_picks=[], time_shift=0)
        saved_data[subj] = {
            "X": X,
            "y": y
        }

def get_epochs_info(subj):
    if subj in saved_epochs_info:
        return saved_epochs_info[subj]
    folder_dict = meg.get_folder_dict()
    subj_aligned = aligned_dir[subj]
    meg_dir = meg_locations[subj]

    epochs, _ = meg.get_processed_meg_data(subj_aligned, folder_dict, meg_dir)
    epochs = epochs.load_data().resample(40).pick_types(meg=True)
    saved_epochs_info[subj] = epochs.info
    return epochs.info


def get_model(model_type):
    if model_type == "logistic":
        return ml.LogisticSlidingModel(max_iter=1500, n_classes=9, k=25, C=0.09, l1_ratio=0.95)
    elif model_type == "svm":
        return ml.SVMSlidingModel(k=25, C=0.85)

def plot_weights(model_type="logistic", n_exp_tests=10, title="model_weights_average", permutation_test=False):
    evoked_pattern_lst = []
    for _ in range(n_exp_tests):
        evoked_pattern_lst_temp = []
        for subj in meg_subj_lst:
            if subj not in saved_data:
                load_data()
            X, y = saved_data[subj]["X"], saved_data[subj]["y"]

            shuffle_idx = np.random.choice(len(X), len(X), replace=False)
            X = X[shuffle_idx]
            y = y[shuffle_idx]

            if permutation_test:
                np.random.shuffle(y)
            
            epochs_info = get_epochs_info(subj)
            model = get_model(model_type)
            evoked_patterns = model.get_patterns(X, y, epochs_info)
            evoked_pattern_lst_temp.append(evoked_patterns)

        evoked_pattern_lst.append(mne.combine_evoked(evoked_pattern_lst_temp, "equal"))

    evoked_patterns_all = mne.combine_evoked(evoked_pattern_lst, "equal")
    evoked_patterns_all.plot_topomap(title="All Weights %s" % model_type, time_unit='s', times=np.arange(0,0.39, 0.025))
    plt.savefig("../Figures/final_results/weights/%s.png" % title)
    plt.clf()






        

def main():
    load_data()
    plot_weights(model_type="logistic", title="logistic_model_weights_average")
    plot_weights(model_type="logistic", permutation_test=True, title="logistic_model_weights_permutation_average")
    
    plot_weights(model_type="svm", title="svm_model_weights_average")
    plot_weights(model_type="svm", permutation_test=True, title="svm_model_weights_permutation_average")


if __name__ == "__main__":
    main()