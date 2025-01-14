# iitkgpnew443
# Install dependencies
!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5
!pip install -r requirements.txt
import os
import shutil

# Define the root dataset path
dataset_root = r"C:\\Users\\Yatindra Rai\\Downloads\\drone_data99"

# Ensure YOLO dataset structure
stages = ['OP1', 'OP2', 'OP3']
for stage in stages:
    for split in ['train', 'val']:
        img_src = os.path.join(dataset_root, stage, 'images', split)
        lbl_src = os.path.join(dataset_root, stage, 'labels', split)
        img_dest = f'datasets/{stage}/images/{split}'
        lbl_dest = f'datasets/{stage}/labels/{split}'

        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)

        shutil.copytree(img_src, img_dest, dirs_exist_ok=True)
        shutil.copytree(lbl_src, lbl_dest, dirs_exist_ok=True)
for stage in stages:
    dataset_yaml = f"""
    train: datasets/{stage}/images/train
    val: datasets/{stage}/images/val

    nc: 1  # Number of classes (saplings)
    names: ['sapling']
    """
    yaml_path = f'datasets/{stage}.yaml'
    with open(yaml_path, 'w') as f:
        f.write(dataset_yaml)
    print(f"Generated {yaml_path}")
for stage in stages:
    dataset_yaml = f"""
    train: datasets/{stage}/images/train
    val: datasets/{stage}/images/val

    nc: 1  # Number of classes (saplings)
    names: ['sapling']
    """
    yaml_path = f'datasets/{stage}.yaml'
    with open(yaml_path, 'w') as f:
        f.write(dataset_yaml)
    print(f"Generated {yaml_path}")
for stage in stages:
    print(f"Training YOLOv5 model for {stage}...")
    !python train.py --img 640 --batch 16 --epochs 50 --data datasets/{stage}.yaml --weights yolov5s.pt --project runs/train --name sapling_detection_{stage}
for stage in stages:
    print(f"Evaluating YOLOv5 model for {stage}...")
    !python val.py --data datasets/{stage}.yaml --weights runs/train/sapling_detection_{stage}/weights/best.pt --img 640
for stage in stages:
    print(f"Running inference for {stage}...")
    !python detect.py --weights runs/train/sapling_detection_{stage}/weights/best.pt --img 640 --source datasets/{stage}/images/val
from PIL import Image
import matplotlib.pyplot as plt
import os

for stage in stages:
    results_path = f'runs/detect/exp_{stage}'
    images = [os.path.join(results_path, img) for img in os.listdir(results_path) if img.endswith('.jpg')]

    print(f"Results for {stage}:")
    for img_path in images:
        img = Image.open(img_path)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
