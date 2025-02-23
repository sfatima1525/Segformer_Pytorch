# This is a main file
# run this file for training


import torch
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
# from segmentation_models_pytorch import Unet
from huggingface_hub import notebook_login
from transformers import TrainingArguments
from transformers import TrainerCallback
from transformers import Trainer
from PIL import Image
import numpy as np
import datasets
import torch
from torch import nn
import evaluate
import matplotlib.pyplot as plt
import copy
from transformers import SegformerForSemanticSegmentation
from datasets import Dataset
import torchvision.transforms.functional as TF
from transformers import pipeline
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from torchvision.transforms import ColorJitter
from transformers import SegformerFeatureExtractor
from sklearn.model_selection import train_test_split

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    cuda_available = torch.cuda.is_available()
print(cuda_available)

class_names = {
    0: "EPI",
    1: "GLD",
    2: "INF",
    3: "RET",
    4: "FOL",
    5: "PAP",
    6: "HYP",
    7: "KER",
    8: "BKG",
    9: "BCC",
    10: "SCC",
    11: "IEC"
}


def train_transforms(example_batch):
    feature_extractor = SegformerFeatureExtractor()
    jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
    print("trace1")
    images = [jitter(x) for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['labels']]
    inputs = feature_extractor(images, labels)
    print("exec")
    print(inputs.size)
    inputs = np.array(inputs)
    return inputs


def val_transforms(example_batch):
    feature_extractor = SegformerFeatureExtractor()
    jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
    images = [jitter(x) for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['labels']]
    inputs = feature_extractor(images, labels)
    inputs = np.array(inputs)
    return inputs


def load_data(image_dir, label_dir, batch_size=500):
    images = []
    labels = []
    count = 1

    image_files = sorted(os.listdir(image_dir))  # Get sorted image files

    print(f"Total image files to process: {len(image_files)}")

    dataset_list = []  # List to hold batches of dataset

    for i in range(0, len(image_files), batch_size):
        batch_image_files = image_files[i:i + batch_size]

        for image_file in batch_image_files:
            image_path = os.path.join(image_dir, image_file)
            image = np.array((Image.open(image_path).convert("RGB")), dtype=np.float32)
            image = np.moveaxis(image, -1, 0)
            images.append(image)

            label_file = image_file.replace(".png", ".png")
            label_path = os.path.join(label_dir, label_file)
            label = (Image.open(label_path).convert("RGB"))
            mask_gray = label.convert("L")
            mask_gray = np.asarray(mask_gray)
            mask_gray = np.copy(mask_gray)

            for x in range(0, (label.size[1])):
                for y in range(0, (label.size[0])):
                    pixel = label.getpixel((y, x))  # Replace (x, y) with the coordinates of the pixel
                    if pixel[0] == 73 and pixel[1] == 0 and pixel[2] == 106:
                        mask_gray[x][y] = 0
                    elif pixel[0] == 108 and pixel[1] == 0 and pixel[2] == 115:
                        mask_gray[x][y] = 1
                    elif pixel[0] == 145 and pixel[1] == 1 and pixel[2] == 122:
                        mask_gray[x][y] = 2
                    elif pixel[0] == 181 and pixel[1] == 9 and pixel[2] == 130:
                        mask_gray[x][y] = 3
                    elif pixel[0] == 216 and pixel[1] == 47 and pixel[2] == 148:
                        mask_gray[x][y] = 4
                    elif pixel[0] == 236 and pixel[1] == 85 and pixel[2] == 157:
                        mask_gray[x][y] = 5
                    elif pixel[0] == 254 and pixel[1] == 246 and pixel[2] == 242:
                        mask_gray[x][y] = 6
                    elif pixel[0] == 248 and pixel[1] == 123 and pixel[2] == 168:
                        mask_gray[x][y] = 7
                    elif pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                        mask_gray[x][y] = 8
                    elif pixel[0] == 127 and pixel[1] == 255 and pixel[2] == 255:
                        mask_gray[x][y] = 9
                    elif pixel[0] == 127 and pixel[1] == 255 and pixel[2] == 142:
                        mask_gray[x][y] = 10
                    elif pixel[0] == 255 and pixel[1] == 127 and pixel[2] == 127:
                        mask_gray[x][y] = 11
                    else:
                        mask_gray[x][y] = 8  # Assign a default class for unknown pixels

            mask_gray = np.array(mask_gray)
            labels.append(mask_gray)
            print(count)
            count += 1
            
        #if count==700:
         # break

        # Create batch dataset
        dataset_dict = {'pixel_values': images, 'labels': labels}
        batch_dataset = datasets.Dataset.from_dict(dataset_dict)
        batch_dataset.set_format(type='numpy', columns=['pixel_values', 'labels'])
        dataset_list.append(batch_dataset)

        # Clear the lists for the next batch
        images = []
        labels = []

    # Concatenate all batch datasets
    dataset = datasets.concatenate_datasets(dataset_list)
    print("Complete loading data")
    return dataset


def calculate_class_distribution(dataset, num_classes):
    all_labels = np.concatenate([sample["labels"].flatten() for sample in dataset])
    class_counts = np.bincount(all_labels, minlength=num_classes)
    total_pixels = all_labels.size
    class_percentage = (class_counts / total_pixels) * 100

    for i in range(num_classes):
        print(f"Class {i} ({class_names[i]}): {class_percentage[i]:.2f}%")


def load_id(path):
    i2l = os.path.join(path, "id2label.json")
    import json
    f = open(i2l, "r")
    # Reading from file
    id2label = json.loads(f.read())
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def compute_metrics(eval_pred):
    with torch.no_grad():
        metric = evaluate.load("evaluate/metrics/mean_iou")
        print("trace1")
        feature_extractor = SegformerFeatureExtractor()
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        print("trace2")
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        # currently using _compute instead of compute
        # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
        metrics = metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=255,
            reduce_labels=feature_extractor.do_reduce_labels,
        )

        # add per category metrics as individual key-value pairs
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
        return metrics


def login_hf():
    notebook_login()


##############################MODELCHECKPOINTCALLBACK######################################################
class ModelCheckpointCallback(TrainerCallback):
    def __init__(self, dirpath, save_top_k=1, monitor="val_loss"):
        self.dirpath = dirpath
        self.save_top_k = save_top_k
        self.monitor = monitor

    def on_epoch_end(self, args, state, control, **kwargs):
        # Save the model checkpoint after each epoch
        print("model saved1")
        model.save_pretrained(self.dirpath)
        print("model saved2")
        return control


# Press the green button in the gutter to run the script.
# if _name_ == '_main_':
root_dir = '/home/sfatima7/sanas_research/project_1/DATASET_10X/10x/'
train_image_dir = '/home/sfatima7/sanas_research/project_1/DATASET_10X/10x/Data_Augmentation/Train_X/New_Train_Patch_Images/'
train_label_dir = '/home/sfatima7/sanas_research/project_1/DATASET_10X/10x/Data_Augmentation/Train_X/New_Train_Patch_Masks/'
val_image_dir = '/home/sfatima7/sanas_research/project_1/DATASET_10X/10x/Data_Augmentation/Train_X/New_Val_Patch_Images/'
val_label_dir = '/home/sfatima7/sanas_research/project_1/DATASET_10X/10x/Data_Augmentation/Train_X/New_Val_Patch_Masks/'

print_hi('PyCharm')
print("                              ")
print("step1 is to load the data")
print("                              ")

# pretrained_model_name = "E:\SanaFatimaMS22Data\Pycharmprojects\pythonProject\checkpoints"
print("now read the id2label file")
print("                              ")
id2label, label2id = load_id(root_dir)
# model = SegformerForSemanticSegmentation.from_pretrained(
#         pretrained_model_name,
#         id2label=id2label,
#         label2id=label2id
# )

print("   ............................Loading training data...")
train_ds = load_data(image_dir=train_image_dir, label_dir=train_label_dir)

print("................................Loading validation data...")
val_ds = load_data(image_dir=val_image_dir, label_dir=val_label_dir)

# Calculate class distribution
print("now Calculate class distribution train data")
calculate_class_distribution(train_ds, num_classes=len(class_names))
# Calculate class distribution
print("now Calculate class distribution of validation data ")
calculate_class_distribution(val_ds, num_classes=len(class_names))

print("Apply transformations")
# Apply transformations
#train_ds = copy.deepcopy(dataset)
#test_ds = copy.deepcopy(dataset)
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)

# changing type to numpy
train_ds.set_format(type='numpy', columns=['pixel_values', 'labels'])
val_ds.set_format(type='numpy', columns=['pixel_values', 'labels'])

# Fine-tune a model
pretrained_model_name = "/home/sfatima7/sanas_research/project_1/nividia-mit-b4/"
#pretrained_model_name = "/home/sfatima7/sanas_research/project_1/testrun/AgainCodeRun/NEWCODE/checkpoints/"
model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    id2label=id2label,
    label2id=label2id
)

# set up the trainer
epochs =50
lr = 0.00006
batch_size = 3

hub_model_id = "segformer-b0-finetuned-segments-sidewalk-2"
training_args = TrainingArguments("Newcode_segformer-b0-finetuned-segments-sidewalk-outputs",
                                  learning_rate=lr,
                                  num_train_epochs=epochs,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  save_total_limit=3,
                                  evaluation_strategy="epoch",
                                  save_strategy="epoch",
                                  save_steps=20,
                                  eval_steps=20,
                                  logging_steps=1,
                                  eval_accumulation_steps=5,
                                  load_best_model_at_end=True,

                                  )

# Hugging Face login
## login_hf()

# Checkpoints
from transformers import SegformerForSemanticSegmentation

checkpoint_dir = '/home/sfatima7/sanas_research/project_1/DATASET_10X/checkpoints/'

os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize the checkpoint callback
checkpoint_callback = ModelCheckpointCallback(dirpath=checkpoint_dir, save_top_k=1, monitor="eval_loss")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[checkpoint_callback],
)
# Load the model from the saved checkpointsmodel = SegformerForSemanticSegmentation.from_pretrained(checkpoint_dir)
print("saved checkpoint")
# Now your model is loaded with the parameters from the saved checkpoints
print("Start the training")
trainer.train()

