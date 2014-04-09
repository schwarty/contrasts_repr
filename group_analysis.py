import os
import glob

import numpy as np
import nibabel as nb

from nipy.modalities.fmri.glm import GeneralLinearModel
from nilearn.input_data import NiftiMasker
from sklearn.preprocessing import LabelBinarizer
from joblib import Memory

cache_dir = '/home/ys218403/Data/cache'
root_dir = '/home/ys218403/Data/intra_stats'
# root_dir = '/media/ys218403/mobile/brainpedia/intra_stats'
result_dir = '/home/ys218403/Data/group_stats'

target = []
niimgs = []
studies = []

for study_dir in glob.glob(root_dir + '/*'):
    study_id = os.path.split(study_dir)[1]
    y = glob.glob(study_dir + '/sub???/*face_vs_house*.nii.gz*')
    # y = glob.glob(study_dir + '/sub???/model/model002/z_maps/*face_vs_house*.nii.gz*')
    if len(y) != 0:
        target.extend([(study_id, )] * len(y))
        niimgs.extend(y)
        studies.append(study_id)

lb = LabelBinarizer()
masker = NiftiMasker(mask='mask.nii.gz',
                     standardize=True,
                     memory=Memory(cache_dir))

X = lb.fit_transform(target)
X[:, -1] = 0

Y = masker.fit_transform(niimgs)

glm = GeneralLinearModel(X)
glm.fit(Y, model='ols')

affine = nb.load('mask.nii.gz').get_affine()

vs_baseline = []
for i, study_id in enumerate(studies[:-1]):
    con_val = np.zeros(len(studies))
    con_val[i] = 1
    contrast = glm.contrast(con_val, contrast_type='t')
    z_map = masker.inverse_transform(contrast.z_score())
    z_map.to_filename(result_dir + '/%s.nii.gz' % study_id)

    # vs_baseline.append()
