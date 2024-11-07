import h5py
import os
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_hdf5_path', type=str, default="datasets/demo.hdf5")
parser.add_argument('--output_folder', type=str, default="hdf5_images")
parser.add_argument('--demo', type=str, default='demo_0')
parser.add_argument('--obs_key', type=str, default='base_camera')

args = parser.parse_args()
os.makedirs(args.output_folder, exist_ok=True)

with h5py.File(args.input_hdf5_path, 'r') as hdf5_file:
    image_dataset = hdf5_file[f'data/{args.demo}/obs/{args.obs_key}']
    
    for i in range(0, image_dataset.shape[0]): 
        image_data = image_dataset[i]
        
        if len(image_data.shape) == 3: 
            image = Image.fromarray(np.uint8(image_data))
            output_file_path = os.path.join(args.output_folder, f'image_{i}.png')
            image.save(output_file_path)

