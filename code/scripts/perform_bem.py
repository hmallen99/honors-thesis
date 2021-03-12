import mne

subjs = [
    "KY",
    "YY",
    "HY",
    "TOu",
]

for s in subjs:
    mne.bem.make_watershed_bem(s, subjects_dir="/usr/local/freesurfer/subjects", overwrite=True)