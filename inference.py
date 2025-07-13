from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import numpy as np
import matplotlib.pyplot as plt
import os
#from agronav_mobile import *

# Config and checkpoint
cwd = os.getcwd()

# Path to config file (.py)
config_file = '/configs/malezas/segformer_mit-b5_malezas.py' #Modificar
config_path = cwd + config_file

# Path to checkpoint file (.pth)
checkpoint_file = '/checkpoints/modelo3.pth' #Modificar
checkpoint_path = cwd + checkpoint_file

# Init model
model = init_model(config_path, checkpoint_path, device='cuda:0')

# Image path.
image_filename = 'demo.jpg' #Modificar
image_path = os.path.join(cwd, image_filename)

# Where will be saved.
output_dir = os.path.join(cwd, 'results')
os.makedirs(output_dir, exist_ok=True)

img = mmcv.imread(image_path, channel_order='rgb')

# Run inference
result = inference_model(model, img)

# Output path for visualization
out_path = os.path.join(output_dir, f'result_{image_filename}')
show_result_pyplot(model, img, result, show=False, save_dir=output_dir)

print("Processing complete. Results saved in:", output_dir)
