# CREATE PATCHES OF SPLITED DATA
# As I have Skin Histology images So I converted the images into 512 overlapping patches

from PIL import Image, ImageOps
#Image.MAX_IMAGE_PIXELS = 933120000
Image.MAX_IMAGE_PIXELS = 1_866_240_000
import os
import base64
import io
from PIL import Image, ImageOps

def blank_image_creation(save_dir):
    #Save the blank image to the folder.
    image_base_64 = ("iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAIAAADTED8xAAADdUlEQVR4Ae3BAQEAAACCoPo"
                     "/2oYkFIj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeFYj0qkCkVwUivSoQ6VWBSK8KRHpVINKrApFeDTai/h+Vjas1AAAAAElFTkSuQmCC")
    image_data = base64.b64decode(image_base_64)
    image_file = io.BytesIO(image_data)
    image = Image.open(image_file)
    image.save(os.path.join(save_dir,"image.jpg"))

def create_patches(image_path, patch_size, temp_path):
    # Load the large image
    print("patch_locations started")

    path_addendum = "input_patches/testA"
    print(image_path)
    image = Image.open(image_path)
    width, height = image.size

    # add white padding to the image of size patch_size/2 along side all directions
    image = ImageOps.expand(image, border=patch_size, fill=(0, 0, 0))
    # show the image

    width, height = image.size
    print(width, height)
    # Create a directory to store the patches
    patch_directory = os.path.join(temp_path, path_addendum)
    os.makedirs(patch_directory, exist_ok=True)
    os.makedirs(os.path.join(temp_path, 'input_patches/testB'), exist_ok=True)

    # Calculate the number of patches in each dimension
    num_patches_width = width // patch_size
    num_patches_height = height // patch_size

    patch_locations = []
    print(patch_directory)
    # Iterate over each patch and save it
    for i in range(0, height - int(patch_size / 2.0), patch_size // 2):
        for j in range(0, width - int(patch_size / 2.0), patch_size // 2):
            # Calculate the coordinates of the current patch
            left = j
            upper = i
            right = left + patch_size
            lower = upper + patch_size
            #print(left, upper, right, lower)
            # Extract the patch from the image
            patch = image.crop((left, upper, right, lower))

            # Generate the patch name
            patch_name = f"{os.path.splitext(os.path.basename(image_path))[0]}.[{left}x{upper}].png"

            # Save the patch
            patch_path = os.path.join(patch_directory, patch_name)
            patch.save(patch_path)
            if left % patch_size == 0 and upper % patch_size == 0:
                patch.save(patch_path)
            elif (left % 512 != 0 and left % 256 == 0) and (upper % 512 != 0 and upper % 256 == 0):
                patch.save(patch_path)


            # Record the patch location
            patch_locations.append(((left, upper), patch_name))

    return patch_locations


input_dir = r'/home/sfatima7/sanas_research/project_1/DATASET_10X/10x/Training_masks/'
output_dir = r'/home/sfatima7/sanas_research/project_1/DATASET_10X/10x/512_train_Y/'

for image in os.listdir(input_dir):

    my_image_path = os.path.join(input_dir, image)
    _ = create_patches(my_image_path, 512,
                   output_dir)

    print('Done')
   
   
