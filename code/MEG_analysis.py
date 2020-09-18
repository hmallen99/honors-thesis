import os
import numpy as np
import mne
import re
#import matplotlib.pyplot as plt

%matplotlib qt

ch_exclude = ['MEG0834','MEG0835','MEG0836','MEG0844','MEG0845','MEG0846','MEG2914','MEG2915','MEG2916','MEG2924','MEG2925',
'MEG2926','MEG2934','MEG2935','MEG2936','MEG2944','MEG2945','MEG2946','MEG3014','MEG3015','MEG3016','MEG3024','MEG3025',
'MEG3026','MEG3034','MEG3035','MEG3036','MEG3044','MEG3045','MEG3046','MEG3114','MEG3115','MEG3116','MEG3124','MEG3125',
'MEG3126','MEG3134','MEG3135','MEG3136','MEG3144','MEG3145','MEG3146','MEG3214','MEG3215','MEG3216','MEG3224','MEG3225',
'MEG3226','MEG3234','MEG3235','MEG3236','MEG3244','MEG3245','MEG3246']

def apply_ica(raw):
    ica = mne.preprocessing.ICA(n_components=20, random_state=5, max_iter=800)
    ica.fit(raw)

    ica.exclude = [0, 1]
    #ica.plot_properties(raw, picks = ica.exclude)
    #ica.plot_components()

    raw.load_data()
    ica_raw = ica.apply(raw)
    return ica_raw

def epoch_data(data):
    #print(raw.ch_names)
    chan_idxs = [data.ch_names.index(ch) for ch in data.ch_names]
    events = mne.find_events(data)
    epochs = mne.Epochs(data, events, event_id=16384, tmin=-0.5, tmax = 1)
    #epochs.plot_image(picks = ['MEG2312'])
    return epochs

def plot_evoked(epochs):
    evoked = epochs["16384"].average()
    evoked.pick_types('grad').plot_topo(color='r')
    # save figure here

def process_data(filename):
    data_path = folder_dict[filename]
    raws = [mne.io.read_raw_fif(raw_file) for raw_file in data_path]
    raw = mne.io.concatenate_raws(raws)

    original = raw.copy()
    raw.load_data().filter(l_freq=2, h_freq=40)
    raw.pick_types(meg="grad", stim=True, exclude = ch_exclude)

    ica_raw = apply_ica(raw)
    epochs = epoch_data(ica_raw)
    plot_evoked(epochs)


def main():
    home = '../../../../MEG_raw01/'
    home_folders = os.listdir(home)
    home_folders = [os.path.join(home, entry) for entry in home_folders if re.findall(r"^\w+", entry)]
    folder_dict = {}
    for folder in home_folders:
        entries = os.listdir(folder)
        folder_dict[folder] = [os.path.join(folder, entry) for entry in entries if re.findall(r"^\w+SD_\w*raw.fif", entry)]

    for file in folder_dict.keys():
        process_data(file)

if __name__ == "__main__":
    main()
