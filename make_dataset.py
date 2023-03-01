import nilearn
import numpy as np
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn.input_data import NiftiLabelsMasker
from pathlib import Path
import os

dataset = datasets.fetch_abide_pcp(data_dir='data/', legacy_format=False)

atlases = datasets.fetch_atlas_basc_multiscale_2015(version='sym', data_dir='data/')
scales = [444, 122]
scales = [f'scale{scale}' for scale in scales]

for scale in scales:
  if scale == 'scale444':
    continue
  atlas = atlases[scale]

  masker = NiftiLabelsMasker(
    labels_img=atlas, 
    standardize=True, 
    memory='nilearn_cache', 
    verbose=1)

  save_folder = Path(f'abide_fc_dataset_{scale}/input/')
  os.makedirs(save_folder, exist_ok=True)

  for i in range(len(dataset.func_preproc)):
    print(f'Processing {i} of {len(dataset.func_preproc)}')
    file_id = dataset.phenotypic.iloc[i]['FILE_ID']
    time_series = masker.fit_transform(dataset.func_preproc[i])
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    np.save(save_folder/f'{file_id}.npy', correlation_matrix)


