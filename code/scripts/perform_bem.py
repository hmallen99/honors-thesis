import mne

subjs = [
    "KY-aligned",
    "YY-aligned",
    "HY-aligned",
    "TOu-aligned",
]

for s in subjs:
    mne.bem.make_watershed_bem(s, subjects_dir="/usr/local/freesurfer/subjects", overwrite=True)