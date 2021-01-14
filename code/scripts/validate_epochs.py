import os
import numpy as np
import mne
import re


home = '../../../../MEG_raw01/'
home_folders = os.listdir(home)
home_folders = [os.path.join(home, entry) for entry in home_folders if re.findall(r"^\w+", entry)]
folder_dict = {}
for folder in home_folders:
    entries = os.listdir(folder)
    folder_dict[folder] = [os.path.join(folder, entry) for entry in entries if re.findall(r"^\w+SD_\w*raw.fif", entry)]

ch_exclude = ['MEG0834','MEG0835','MEG0836','MEG0844','MEG0845','MEG0846','MEG2914','MEG2915','MEG2916','MEG2924','MEG2925',
'MEG2926','MEG2934','MEG2935','MEG2936','MEG2944','MEG2945','MEG2946','MEG3014','MEG3015','MEG3016','MEG3024','MEG3025',
'MEG3026','MEG3034','MEG3035','MEG3036','MEG3044','MEG3045','MEG3046','MEG3114','MEG3115','MEG3116','MEG3124','MEG3125',
'MEG3126','MEG3134','MEG3135','MEG3136','MEG3144','MEG3145','MEG3146','MEG3214','MEG3215','MEG3216','MEG3224','MEG3225',
'MEG3226','MEG3234','MEG3235','MEG3236','MEG3244','MEG3245','MEG3246']


bad_epochs = []

for filename in folder_dict.keys():
    datapath = folder_dict[filename]
    for raw_file in datapath:
        raw = mne.io.read_raw_fif(raw_file)
        raw.load_data().filter(l_freq=2, h_freq=40)
        raw.pick_types(meg="grad", stim=True, exclude = ch_exclude) 

        events = mne.find_events(raw)
        epochs = mne.Epochs(raw, events, event_id=16384, tmin=-0.5, tmax = 1)

        if len(epochs.events) != 100:
            bad_epochs += [raw_file]
        
        del raw


text_file = open("bad_epoch_files.txt", "w")
for b in bad_epochs:
    text_file.write(b)
    text_file.write("\n")

text_file.close()