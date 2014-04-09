import os
import glob

import numpy as np
import nibabel as nb

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from nilearn.input_data import NiftiMasker
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from joblib import Memory

cache_dir = '/home/ys218403/Data/cache'
root_dir = '/home/ys218403/Data/intra_stats'
# root_dir = '/media/ys218403/mobile/brainpedia/intra_stats'
result_dir = '/home/ys218403/Data/group_stats'

target = []
niimgs = []

for study_dir in glob.glob(root_dir + '/*'):
    study_id = os.path.split(study_dir)[1]
    house = glob.glob(study_dir + '/sub???/*house_vs_baseline*.nii.gz')
    face = glob.glob(study_dir + '/sub???/*face_vs_baseline*.nii.gz')
    niimgs.extend(house)
    niimgs.extend(face)

    target.append([(study_id, '0house')] * len(house))
    target.append([(study_id, '1face')] * len(face))

target = np.vstack(target)

masker = NiftiMasker(mask='mask.nii.gz',
                     standardize=True,
                     memory=Memory(cache_dir))
selector = SelectPercentile(f_classif, percentile=50)
loader = Pipeline([('masker', masker), ('selector', selector)])

le = LabelEncoder()
y = le.fit_transform(target[:, 1])

X = loader.fit_transform(niimgs, y)


clf = LogisticRegression()

cv = StratifiedKFold(y=y[target[:, 0] == 'ds105'], n_folds=5)

# clf.fit(X[target[:, 0] == 'pinel2009twins'], y[target[:, 0] == 'pinel2009twins'])

# print clf.score(X[target[:, 0] != 'pinel2009twins'], y[target[:, 0] != 'pinel2009twins'])

# scores = cross_val_score(
#     clf,
#     X[target[:, 0] == 'ds105'],
#     y[target[:, 0] == 'ds105'], cv=cv, n_jobs=-1)
# print scores
clf.fit(X, y)
coef = loader.inverse_transform(clf.coef_)
coef.to_filename('/tmp/coef.nii.gz')
