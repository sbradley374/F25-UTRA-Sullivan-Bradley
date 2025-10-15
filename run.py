import json
import numpy as np

from FOD.Predictor import Predictor
from FOD.dataset import AutoFocusDataset
 

with open('config.json') as f:
    config = json.load(f)
np.random.seed(config['General']['seed'])

list_data = config['Dataset']['paths']['list_datasets']
autofocus_datasets_test = [
    AutoFocusDataset(config, dataset_name, 'test') for dataset_name in list_data
]

test_image_paths = [p for ds in autofocus_datasets_test for p in ds.paths_images]
predictor = Predictor(config, test_image_paths)
predictor.run()
