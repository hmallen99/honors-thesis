# MEG and MRI analysis of Serial Dependence

Author: Henry Allen

## Directories:


### Code
IMPORTANT!: If you get errors, they most likely result from a missing or incorrect file path. This is especially likely when trying to generate figures or trying to save data in the Data/ folder.

Contains all the code for the project. 

#### Main Folder
1. `bayesian_model`: This code is not complete, but is meant to implement the vanBergen modification of the IEM
2. `DoG`: Contains permutation tests for derivative of Gaussians
3. `file_lists`: Contains various python lists and dictionaries for making automation easier. This is a bad way to do things, and these lists should probably be stored in .mat or .npy files in the future
4. `inverted_encoding_model`: Completed Inverted encoding model, adapted from Sprague code
5. `load_data`: various functions to help load MEG data
6. `machine_learning`: various machine learning classes and helper functions
7. `MEG_analysis`: meg preprocessing code
8. `permutation_test`: main code for running permutation tests with MNE decoding algorithms
9. `source_localization`: source localization helper functions

#### Jupyter Notebooks

- These notebooks are just my initial explorations for MEG analysis and decoding analysis.

- The Adjustment_Analysis6.ipynb code does not belong to me (Henry)

- Move these notebooks to `code/` if you want them to work properly

#### Old Analysis Files

The work in here is either confusing to read, or may not work. Enter at your own risk

1. `circular_regression`: attempts to do circular regression, but wasn't successful
2. `main_analysis`: Initial analysis to run machine learning models
3. `test_decoding_sample_rate`: test decoding with higher sample rate
4. `serial_dependence`: various functions for analyzing serial dependence, many of which were unsuccessful

#### Scripts

This file includes some scripts to make it easier to do analysis. You may need to move them to the main /code/ directory
for them to work

1. `perform_bem`: Runs the make_bem_watershed code for all subjects in "subjs"
2. `plot_meg_figs`: Makes figures with averaged evoked responses and source localization figures
3. `plot_model_weights`: plots the MVPA patterns for MNE decoding models
4. `save_mat`: saves loaded data to a mat file so that it can be easily reused
5. `validate_epochs`: Checks the alignment of stim tracks with meg epochs


### Figures

IMPORTANT!: Before running the main analysis code, you will have to set the path for where to save figures, as the current paths may not exist. 

1. `initial_results`/: Figures generated from exploratory analysis and preprocessing. Results here are not necessarily indicative of final model performance

2. `intermediate_results`/: Figures from intermediate data processing and analysis after preprocessing. There is a lot of clutter here, and these files are just kept for archiving progress

3. `final_results`/: Results computed with permutation tests, used in final draft of thesis and VSS poster.


### Data
Contains saved pre-processed data so that we don't have to run preprocessing every time we want to run the decoding analysis


## Pre-processsing Overview
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