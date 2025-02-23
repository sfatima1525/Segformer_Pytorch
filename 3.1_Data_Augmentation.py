
# Remove background-only images and labels
# I have class imbalance data, So use a data augmentation technique to handle this. 
# First remove extra background images/labels, then apply the data augmentation technique. 

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Define the background color
background_color = (0, 0, 0)  # RGB color for background

def is_background_only(mask, background_color=(0, 0, 0), tolerance=10):
    # Check if the mask contains only the background color, allowing a tolerance
    return np.all(np.abs(mask - background_color) <= tolerance)

def remove_background_images_and_labels(image_dir, label_dir, image_save_dir, label_save_dir, background_color=(0, 0, 0)):
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)

    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))

    for image_file, label_file in tqdm(zip(image_files, label_files), desc="Removing background-only images and labels", total=len(image_files)):
        if not image_file.endswith('.png') or not label_file.endswith('.png'):
            continue

        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")

        label_array = np.array(label)

        # Check if the label is not background-only
        if not is_background_only(label_array, background_color):
            # Save the image and label only if the label is not background-only
            image.save(os.path.join(image_save_dir, image_file))
            label.save(os.path.join(label_save_dir, label_file))

# Directory paths
image_dir = '/home/sfatima7/sanas_research/project_1/DATASET_10X/10x/512_train_X/input_patches/testA/'
label_dir = '/home/sfatima7/sanas_research/project_1/DATASET_10X/10x/512_train_Y/input_patches/testA/'
image_save_dir = '/home/sfatima7/sanas_research/project_1/DATASET_10X/10x/Data_Augmentation/Train_X/bkg_remove/Image/'
label_save_dir = '/home/sfatima7/sanas_research/project_1/DATASET_10X/10x/Data_Augmentation/Train_X/bkg_remove/Label/'

# Remove background-only images and labels
remove_background_images_and_labels(image_dir, label_dir, image_save_dir, label_save_dir)