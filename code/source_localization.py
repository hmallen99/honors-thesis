import mne

def create_source_space(name, fdir, save=False):
    """ 
    Directions to create the freesurfer file
    $ my_subject=sample
    $ my_NIfTI=/path/to/NIfTI.nii.gz
    $ recon-all -i $my_NIfTI -s $my_subject -all
    """
    src = mne.setup_source_space(name, subjects_dir=fdir, spacing='oct6')
    if save:
        src.save("../Data/SourceSpaces/%s-src.fif" % name)
    return src

def get_source_space(fpath):
    return mne.read_source_spaces(fpath)

# run in a python terminal
# mne.bem.make_watershed_bem(name, subjects_dir=fdir, overwrite=True)
def make_bem(name, fdir, save=False):
    model = mne.make_bem_model(name, fdir)
    bem = mne.make_bem_solution(model)
    if save:
        mne.write_bem_solution("%s-bem-sol.fif" % name, bem, overwrite=True)
    return bem

def read_bem(fpath):
    return mne.read_bem_solution(fpath)

def make_forward_sol(evoked, src, bem):
    return mne.make_forward_solution(evoked.info, None, src, bem)

def make_inverse_operator(evoked, forward_sol, cov):
    return mne.minimum_norm.make_inverse_operator(evoked.info, forward_sol, cov, loose=0.2, dpeth=0.8)

def apply_inverse(evoked, inverse_op):
    stc, residual = mne.minimum_norm.apply_inverse(evoked, inverse_op, method="dSPM", pick_ori=None, return_residual=True, verbose=True)
    return stc, residual



