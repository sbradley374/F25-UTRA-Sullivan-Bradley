import json
from glob import glob
from FOD.Predictor import Predictor

with open('config.json', 'r') as f:
    config = json.load(f)

# Run inference on selected training examples
# TODO: Proper train/val/test splits needed
input_images = glob('data/synthetic_data/noisy_512/*00.png')
predictor = Predictor(config, input_images)
predictor.run()
