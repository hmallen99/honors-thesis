# MEG and MRI analysis of Serial Dependence

Author: Henry Allen

## Directories:
1. Code - Contains all of the jupyter notebooks and python scripts for the code. Primary analysis will be done in main_analysis.py, with preprocessing work done in 
MEG_analysis.py and source localization done in source_localization.py

2. Data - Store computed fif files for source localization, evoked responses, etc.

3. Figures - Plots and results


## Pre-processsing Steps

For pre-processing, there are two main parts: (1) Filtering and processing the MEG data and (2) using MRI data for source localization. These steps can be seen as a diagram here:
https://mne.tools/stable/overview/cookbook.html

### MEG pre-processing
1. Import the raw data

2. Filter the data. I used a high-pass filter at 2hz and a low-pass filter at 40hz.

3. Execute Independent Component Analysis (ICA). ICA helps us to remove heartbeats and eye blinks from the signals, which cause artifacts. These bad ICA components were picked out and excluded from the data. ICA also helps with some initial source localization.

4. Epoch the data. In this step, we create a ~2 second window for each trial and time-lock each trial to the stimulus onset. We drop trials that have a peak-to-peak gradiometer amplitude of 4000e-13 T/m.

5. Average the date. Use the mne.Evoked class to average the data and produce ERFs. We also use this data to plot power-spectral density plots, and as an input for source localization.

### Source Localization
1. Generate Freesurfer surfaces

2.  Run `mne.bem.make_watershed_bem()` on the subject to generate a head model

2. Align the mri data with the meg data using `mne coreg`. This will create a new "subject" that you should use for further steps.

3. Run `mne.bem.make_watershed_bem()` on the newly aligned model

4. Create a source space from the Freesurfer data

5. Generate a BEM from the Freesurfer data

6. Calculate the forward solution

7. Calculate the inverse operator

8. Create the source estimate


### Machine Learning
1. 