import nilearn
import numpy as np
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn.input_data import NiftiLabelsMasker

dataset = datasets.fetch_abide_pcp(data_dir='data/', legacy_format=False)

atlas = datasets.fetch_atlas_basc_multiscale_2015(version='sym', data_dir='data/')
atlas = atlas.scale444

masker = NiftiLabelsMasker(
  labels_img=atlas, 
  standardize=True, 
  memory='nilearn_cache', 
  verbose=1)

for i in range(len(dataset.func_preproc)):
  print(f'Processing {i} of {len(dataset.func_preproc)}')
  file_id = dataset.phenotypic.iloc[i]['FILE_ID']
  time_series = masker.fit_transform(dataset.func_preproc[i])
  correlation_measure = ConnectivityMeasure(kind='correlation')
  correlation_matrix = correlation_measure.fit_transform([time_series])[0]
  np.save(f'abide_fc_dataset/input/{file_id}.npy', correlation_matrix)


