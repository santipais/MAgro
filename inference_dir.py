from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import numpy as np
import matplotlib.pyplot as plt
import os

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

# Directory of images to inference.
images = 'data/malezas/images/test'
image_dir = os.path.join(cwd, images)

# Directory where results will be saved.
outputs = 'visualizar' #Modificar
output_dir = os.path.join(cwd, outputs)
os.makedirs(output_dir, exist_ok=True)

# Process each image in the directory
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(image_dir, filename)
        img = mmcv.imread(img_path, channel_order='rgb')

        # Run inference
        result = inference_model(model, img)

        # Output path for visualization
        out_path = os.path.join(output_dir, f'{filename}')
        show_result_pyplot(model, img, result, out_file=out_path, show=False)

print("Processing complete. Results saved in:", output_dir)
