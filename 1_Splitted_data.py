
#STEP 1 IS TO SPLIT THE DATA INTO TRAINING, VALIDATION AND TESTING

import os
import shutil
import random

def split_data(image_dir: str, mask_dir: str, train_ratio: float, val_ratio: float):
    # Define the directories to save training, validation, and testing data
    train_image_dir = os.path.join(base_dir, "Training_images")
    train_mask_dir = os.path.join(base_dir, "Training_masks")
    val_image_dir = os.path.join(base_dir, "Validation_images")
    val_mask_dir = os.path.join(base_dir, "Validation_masks")
    test_image_dir = os.path.join(base_dir, "Testing_images")
    test_mask_dir = os.path.join(base_dir, "Testing_masks")

    # Create directories if they don't exist
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)
    os.makedirs(test_mask_dir, exist_ok=True)

    # Create lists to hold file names
    image_files = os.listdir(image_dir)
    mask_files = os.listdir(mask_dir)

    # Ensure masks are correctly paired with images
    paired_files = []
    for image_file in image_files:
        image_name, image_ext = os.path.splitext(image_file)
        corresponding_mask = None

        for mask_file in mask_files:
            mask_name, mask_ext = os.path.splitext(mask_file)
            if image_name == mask_name:
                corresponding_mask = mask_file
                break

        if corresponding_mask:
            paired_files.append((image_file, corresponding_mask))
        else:
            print(f"No matching mask found for image: {image_file}")

    if not paired_files:
        print("No image-mask pairs found. Check if filenames match between images and masks.")
        return

    random.shuffle(paired_files)

    # Calculate the number of samples for training, validation, and testing
    num_total_samples = len(paired_files)
    num_train_samples = int(num_total_samples * train_ratio)
    num_val_samples = int(num_total_samples * val_ratio)
    num_test_samples = num_total_samples - num_train_samples - num_val_samples

    # Copy training data to the training directories
    for i in range(num_train_samples):
        image_src = os.path.join(image_dir, paired_files[i][0])
        mask_src = os.path.join(mask_dir, paired_files[i][1])

        image_dst = os.path.join(train_image_dir, paired_files[i][0])
        mask_dst = os.path.join(train_mask_dir, paired_files[i][1])

        shutil.copy(image_src, image_dst)
        shutil.copy(mask_src, mask_dst)

    print("Training images have been saved to:", train_image_dir)
    print("Training masks have been saved to:", train_mask_dir)

    # Copy validation data to the validation directories
    for i in range(num_train_samples, num_train_samples + num_val_samples):
        image_src = os.path.join(image_dir, paired_files[i][0])
        mask_src = os.path.join(mask_dir, paired_files[i][1])

        image_dst = os.path.join(val_image_dir, paired_files[i][0])
        mask_dst = os.path.join(val_mask_dir, paired_files[i][1])

        shutil.copy(image_src, image_dst)
        shutil.copy(mask_src, mask_dst)

    print("Validation images have been saved to:", val_image_dir)
    print("Validation masks have been saved to:", val_mask_dir)

    # Copy testing data to the testing directories
    for i in range(num_train_samples + num_val_samples, len(paired_files)):
        image_src = os.path.join(image_dir, paired_files[i][0])
        mask_src = os.path.join(mask_dir, paired_files[i][1])

        image_dst = os.path.join(test_image_dir, paired_files[i][0])
        mask_dst = os.path.join(test_mask_dir, paired_files[i][1])

        shutil.copy(image_src, image_dst)
        shutil.copy(mask_src, mask_dst)

    print("Testing images have been saved to:", test_image_dir)
    print("Testing masks have been saved to:", test_mask_dir)

# Assign directories where your image and mask patches are located
base_dir = "/home/sfatima7/sanas_research/project_1/DATASET_10X/10x"
image_dir = os.path.join(base_dir, "Images")
mask_dir = os.path.join(base_dir, "Masks")

# Define the ratios of data to be used for training, validation, and testing
train_ratio = 0.8  # 80% of data for training
val_ratio = 0.1    # 10% of data for validation

# Split the data
split_data(image_dir, mask_dir, train_ratio, val_ratio)