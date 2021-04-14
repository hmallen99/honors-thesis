import os
import numpy as np
import mne
import re
import matplotlib.pyplot as plt
from file_lists import meg_locations


ch_exclude = ['MEG0834','MEG0835','MEG0836','MEG0844','MEG0845','MEG0846','MEG2914','MEG2915','MEG2916','MEG2924','MEG2925',
'MEG2926','MEG2934','MEG2935','MEG2936','MEG2944','MEG2945','MEG2946','MEG3014','MEG3015','MEG3016','MEG3024','MEG3025',
'MEG3026','MEG3034','MEG3035','MEG3036','MEG3044','MEG3045','MEG3046','MEG3114','MEG3115','MEG3116','MEG3124','MEG3125',
'MEG3126','MEG3134','MEG3135','MEG3136','MEG3144','MEG3145','MEG3146','MEG3214','MEG3215','MEG3216','MEG3224','MEG3225',
'MEG3226','MEG3234','MEG3235','MEG3236','MEG3244','MEG3245','MEG3246']

ica_exclude_picks = [
    [0, 4],
    [0, 4],
    [0, 1],
    [0, 9, 19],
    [0, 1],
    [0, 8],
    [0, 9],
    [0, 1, 2],
    [0],
    [0, 1],
    [3],
    [1, 7],
    [0, 1, 4],
    [3, 9],
    [0, 1, 2, 4],
    [0, 13],
    [0],
    [0, 7],
    [0, 15],
    [0, 3],
    [0]
]

def apply_ica(raw, participant):
    print("Executing ICA")
    ica = mne.preprocessing.ICA(n_components=20, random_state=5, max_iter=800)
    ica.fit(raw)
    ica.exclude = ica_exclude_picks[participant]
    raw.load_data()
    ica_raw = ica.apply(raw)
    return ica_raw

def epoch_data(data, participant=-1, plot=False, bads=[]):
    print("Epoching Data")
    chan_idxs = [data.ch_names.index(ch) for ch in data.ch_names]
    reject = dict(grad=4000e-13)
    events = mne.find_events(data, stim_channel="STI015")
    if len(events) == 806:
        events = mne.find_events(data, stim_channel="STI016")
    epochs = mne.Epochs(data, events, tmin=-0.5, tmax = 1, reject = reject)
    if len(events) == 801:
        epochs.drop(indices=248)
    epochs.drop_bad()
    if plot:
        epochs.plot_drop_log(show=False).savefig("../figures/bad_epochs/bad_epochs_%i" % participant)
    return epochs

def plot_evoked(epochs, participant, plot=False):
    print("Plotting Evoked")
    evoked = epochs["16384"].average()
    title = "Participant %i" % participant
    if plot:
        evoked.pick_types('grad').plot_topo(color='r', title=title, show=False).savefig("../figures/topomap/erf_map_%i.pdf" % participant)
        evoked.plot(picks=["MEG1923"], window_title=title, show=False).savefig("../figures/left_occipital/left_occipital_%i.pdf" % participant)
        evoked.plot(picks=["MEG2332"], window_title=title, show=False).savefig("../figures/right_occipital/right_occipital_%i.pdf" % participant)
    return evoked

def get_raw(data):
    print([raw_file for raw_file in data])
    raws = [mne.io.read_raw_fif(raw_file, verbose=False) for raw_file in data]
    raw = mne.io.concatenate_raws(raws)
    raw.load_data().filter(l_freq=2, h_freq=200)
    raw.pick_types(meg="grad", stim=True, eog=True, exclude = ch_exclude)
    raw.notch_filter(np.arange(60, 241, 60))
    return raw

def process_data(data, participant):
    raw = get_raw(data)
    ica_raw = apply_ica(raw, participant)
    epochs = epoch_data(ica_raw, participant)
    evoked = plot_evoked(epochs, participant)
    return evoked, epochs, ica_raw.info

def get_folder_dict():
    home = '../../../../MEG_raw01/'
    home_folders = os.listdir(home)
    home_folders = [os.path.join(home, entry) for entry in home_folders if re.findall(r"^\w+", entry)]
    folder_dict = {}
    for folder in home_folders:
        entries = os.listdir(folder)
        folder_dict[folder] = [os.path.join(folder, entry) for entry in entries if re.findall(r"^\w+SD_\w*raw.fif", entry)]
    return folder_dict


def get_processed_meg_data(subj, folder_dict, meg_dir):
    epochs_path = "../Data/Epochs3/%s-epo.fif" % subj
    evoked_path = "../Data/Evoked3/%s-ave.fif" % subj
    if os.path.isfile(epochs_path) and os.path.isfile(evoked_path):
        epochs = mne.read_epochs(epochs_path)
        evoked = mne.read_evokeds(evoked_path)[0]
        return epochs, evoked
    else:
        evoked, epochs, info = process_data(folder_dict[meg_dir], 0)
        evoked.save('../Data/Evoked3/%s-ave.fif' % subj)
        epochs.save('../Data/Epochs3/%s-epo.fif' % subj)
        return epochs, evoked


def main():
    folder_dict = get_folder_dict()

    i = 0
    evoked_list = []
    for filename in folder_dict.keys():
        print("Processing file %s" % filename)
        evoked, ep, info = process_data(folder_dict[filename], i)
        evoked.interpolate_bads(reset_bads=False)
        evoked_list.append(evoked)
        i += 1

    #all_evoked = mne.combine_evoked(evoked_list, 'equal')
    #all_evoked.pick_types('grad').plot_topo(color='r', title="erf_average_topo", show=False).savefig("erf_map_average.pdf")
    #all_evoked.plot(picks=["MEG1923"], window_title="left_occipital_average", show=False).savefig("left_occipital_average.pdf")
    #all_evoked.plot(picks=["MEG2332"], window_title="right_occipital_average", show=False).savefig("right_occipital_average.pdf")

if __name__ == "__main__":
    main()

