# Code

Contains all the code for the project

## Main Folder
1. bayesian_model: This code is not complete, but is meant to implement the vanBergen modification of the IEM
2. DoG: Contains permutation tests for derivative of Gaussians
3. file_lists: Contains various python lists and dictionaries for making automation easier. This is a bad way to do things, and these lists should probably be stored in .mat or .npy files in the future
4. inverted_encoding_model: Completed Inverted encoding model, adapted from Sprague code
5. load_data: various functions to help load MEG data
6. machine_learning: various machine learning classes and helper functions
7. MEG_analysis: meg preprocessing code
8. permutation_test: main code for running permutation tests with MNE decoding algorithms
9. serial_dependence: various functions for analyzing serial dependence, many of which were unsuccessful
10. source_localization: source localization helper functions

## Jupyter Notebooks

These notebooks are just my initial explorations for MEG analysis and decoding analysis.

The Adjustment_Analysis6.ipynb code does not belong to me (Henry)

Move these notebooks to `code/` if you want them to work properly

## Old Analysis Files

The work in here is either confusing to read, or may not work. Enter at your own risk

1. circular_regression: attempts to do circular regression, but wasn't successful
2. main_analysis: Initial analysis to run machine learning models
3. test_decoding_sample_rate: test decoding with higher sample rate

## Scripts

This file includes some scripts to make it easier to do analysis. You may need to move them to the main /code/ directory
for them to work

1. perform_bem: Runs the make_bem_watershed code for all subjects in "subjs"
2. plot_meg_figs: Makes figures with averaged evoked responses and source localization figures
3. plot_model_weights: plots the MVPA patterns for MNE decoding models
4. save_mat: saves loaded data to a mat file so that it can be easily reused
5. validate_epochs: Checks the alignment of stim tracks with meg epochs