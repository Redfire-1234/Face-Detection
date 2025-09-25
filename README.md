# Face-Detection
!nvidia-smi # TO SEE THAT YOU ARE CONNECTED TO GPU

!pip install ultralytics # ULTRALYTICS MODULE FOR PRETRAINED WHEIGHTS OF YOLOV8 & YOLOV5

!unzip -q data.zip.zip -d data # UNZIPPING THE DATASET

# COUNT THE NUMBER OF IMAGES IN THE DATASET
import os

folder='data/images'
count=len(os.listdir(folder))
print(count)

# CREATE FOLDER LABELS INSIDE THE DATA FOLDER
import os

os.makedirs("data/labels", exist_ok=True)

# READ THE CSV FILE
import pandas as pd

df = pd.read_csv("data/faces.csv")
print(df.columns)
print(df.head(10))

# CONVERT BOUNDING BOX ANNOTATIONS FROM CSV FILE INTO YOLO FORMAT LABELS
import pandas as pd

df = pd.read_csv("data/faces.csv")

for index, row in df.iterrows():
    # Image dimensions from CSV
    img_w = row['width']
    img_h = row['height']

    # YOLO format
    x_center = ((row['x0'] + row['x1']) / 2) / img_w
    y_center = ((row['y0'] + row['y1']) / 2) / img_h
    bbox_w = (row['x1'] - row['x0']) / img_w
    bbox_h = (row['y1'] - row['y0']) / img_h

    # Save label file
    label_file = os.path.join("data/labels", row['image_name'].replace(".jpg", ".txt"))
    with open(label_file, "a") as f:
        f.write(f"0 {x_center} {y_center} {bbox_w} {bbox_h}\n")

# LIST FIRST 10 IMAGES FROM DATA FOLDER
!ls /content/data/images |head -10

import os
import random
from shutil import copyfile

# SETS DATASET PATH
image_folder = 'data/images'
label_folder = 'data/labels'

# SHUFFLES IMMAGE RAMDOMLY
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
random.shuffle(image_files)

# SPLIT DATASET INTO TRAIN(80%) AND VALIDATION(20%)
train_split_ratio = 0.8
train_count = int(len(image_files) * train_split_ratio)
train_files = image_files[:train_count]
val_files = image_files[train_count:]

# CREATE NEW FOLDERS FOR TRAIN AND VAL SET
os.makedirs('data/train/images', exist_ok=True)
os.makedirs('data/train/labels', exist_ok=True)
os.makedirs('data/val/images', exist_ok=True)
os.makedirs('data/val/labels', exist_ok=True)

# COPY FILSE INTO RESPECTIVE DIRECTORIES
for file in train_files:
    name = file.replace('.jpg', '')
    copyfile(os.path.join(image_folder, file), os.path.join('data/train/images', file))
    copyfile(os.path.join(label_folder, name + '.txt'), os.path.join('data/train/labels', name + '.txt'))

for file in val_files:
    name = file.replace('.jpg', '')
    copyfile(os.path.join(image_folder, file), os.path.join('data/val/images', file))
    copyfile(os.path.join(label_folder, name + '.txt'), os.path.join('data/val/labels', name + '.txt'))

print(f"Training images: {len(train_files)}")
print(f"Validation images: {len(val_files)}")

import os
# GENERATING DATA.YAML FILE FOR WHICH YOLO NEEDS TO KNOW WHERE THE DATASET IS AND WHAT CLASSES TO DETECT

# DATASET PATHS DEFINE
train_images_path = '../data/train/images'
val_images_path = '../data/val/images'

# NUMBER OF CLASS AND CLASS NAME
num_classes = 1
class_names = ['face']

# CREATE DATA YAML CONTENT
data_yaml_content = f"""
path: ../data  # Dataset root directory
train: {train_images_path}  # Training images directory
val: {val_images_path}  # Validation images directory

nc: {num_classes}  # Number of classes
names: {class_names}  # Class names
"""

# SAVE IT AS DATA.YAML
with open("data.yaml", "w") as f:
    f.write(data_yaml_content)

print("data.yaml created successfully!")

from ultralytics import YOLO

# LOAD A PRETRAINED YOLOV8N MODEL
model = YOLO('yolov8s.pt')

# TRAIN THE MODEL USING DATA.YAML FILE FOR 50 EPOCHS
results = model.train(data='data.yaml', epochs=50, imgsz=320)

# EVALUATE THE MODEL ON VALIDATION SET
results = model.val()
print(results)

from ultralytics import YOLO

# runs/detect/train/weights/best.pt THIS IS THE PATH YOLO AUTOMATICALLY SAVES AFTER TRAINING
model = YOLO('runs/detect/train/weights/best.pt') # MODEL IS NOW YOUR TRAINED YOLO MODEL

import os
import shutil

# PATH OF BEST.PT FILE
source_path = 'runs/detect/train/weights/best.pt'
destination_folder = 'my_model1' # CREATE A NEW FOLDER
destination_path = os.path.join(destination_folder, 'my_model1.pt') # SAVE YOUR MODEL AS MY_MODEL1.PT INSIDE THE FOLDER YOU JUST CREATED

# CREATE THE DESTINATION FOLDER
os.makedirs(destination_folder, exist_ok=True)

# MOVE YOUR FILE
shutil.move(source_path, destination_path)

print(f"Model weights moved to {destination_path}")

# ZIP YOUR FILE AND THEN DOWNLOAD IT FROM SIDE BAR
!zip -r my_model.zip my_model1

# UNZIP MY_MODEL
!unzip /content/my_model.zip -d /content/my_model/

# INSTALL ULTRALYTICS MODULE
!pip install ultralytics

# FIRST UPLOAD YOUR TEST_IMAGES FOLDER ON WHICH YOU HAVE TO TEST YOUR YOLO MODEL
# UNZIP YOUR TEST_IMAGES FOLDER
!unzip /content/test_images.zip -d /content/test_images/

# CONTENT IN THE FOLDER
!ls /content/test_images/test_images

results = model.predict(
    source='/content/test_images/test_images',
    save=True,
    show=True
)
#model.predict() → runs inference (prediction) with the YOLO model.

#source='/content/test_images/test_images' → path to your test images.

#save=True → saves the images with bounding boxes drawn (in runs/detect/predict/).

#show=True → displays the results in Colab output cells.

#results → stores prediction results (bounding boxes, class IDs, confidences, etc.).

# DISPLAY YOUR IMAGES IN COLAB NOTEBOOK
from IPython.display import Image, display
import os

output_folder = 'runs/detect/predict/'
for img_file in os.listdir(output_folder):
    if img_file.endswith(('.jpg', '.png')):
        display(Image(filename=os.path.join(output_folder, img_file)))

# ZIP YOUR PREDICT IMAGES IN PREDICTIONS FOLDER
# DOWNLOAD THE ZIP PREDICTIONS FOLDER FROM SIDE BAR
!zip -r predictions.zip runs/detect/predict/

from ultralytics import YOLO

# LOAD THE YOLOV5 TRAINED MODEL
model = YOLO('yolov5s.pt')  # You can choose from yolov5nu.pt, yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.pt

# TRAIN THE MODEL USING DATA.YAML FILE WHICH WAS CREATED EARLIER
result_v5 = model.train(data='data.yaml', epochs=50, imgsz=320)

# EVALUATE THE MODEL ON VALIDATION SET
# Computes metrics like:
# Precision (P)
# Recall (R)
# mAP@0.5
# mAP@0.5:0.95

results = model.val()
print(results)

from ultralytics import YOLO

# runs/detect/train3/weights/best.pt THIS IS THE PATH YOLO AUTOMATICALLY SAVES AFTER TRAINING
model_v5 = YOLO('runs/detect/train3/weights/best.pt') # MODEL_V5 IS NOW YOUR TRAINED YOLO MODEL

import os
import shutil

# Define the paths
source_path = 'runs/detect/train3/weights/best.pt' # TRAINED MODEL FILE YOLO CREATED AFTER TRAINING
destination_folder = 'my_model1_v5' # CREATE NEW FOLDER
destination_path = os.path.join(destination_folder, 'my_model1_v5.pt') # SAVE YOUR MODEL MY_MODEL1_V5 INSIDE THE NEW FOLDER YOU CREATED

# CREATE DESTINATION FOLDER
os.makedirs(destination_folder, exist_ok=True)

# MOVE AND RENAME THE FOLDER
shutil.move(source_path, destination_path)

print(f"Model weights moved to {destination_path}")

# ZIP YOUR MODEL MY_MODEL1_V5 AS MY_MODEL_V5
!zip -r my_model_v5.zip my_model1_v5

# UNZIP THE FOLDER MY_MODEL_V5
!unzip /content/my_model_v5.zip -d /content/my_model_v5/

!ls /content/my_model_v5/my_model1_v5

!pip install ultralytics

results_v5 = model.predict(
    source='/content/test_images/test_images',  # inner folder with images
    save=True,
    show=True
)
#model.predict() → runs inference (prediction) with the YOLO model.
#source='/content/test_images/test_images' → path to your test images.
#save=True → saves the images with bounding boxes drawn (in runs/detect/predict/).
#show=True → displays the results in Colab output cells.
#results → stores prediction results (bounding boxes, class IDs, confidences, etc.).

from IPython.display import Image, display
import os

output_folder = 'runs/detect/train33/'
for img_file in os.listdir(output_folder):
    if img_file.endswith(('.jpg', '.png')):
        display(Image(filename=os.path.join(output_folder, img_file)))
# runs/detect/train33/ → YOLO automatically saves prediction outputs here.
# display(Image(...)) → shows each image inside your notebook.

# zip folder as predictions_v5
!zip -r predictions_v5.zip runs/detect/train33/

# UNZIP YOUR VIDEO FILE
!unzip video4.zip -d unzipped_video4

!pip install ultralytics


!unzip my_model.zip -d my_model


from ultralytics import YOLO

# LOAD YOUR TRAINED YOLOv5 MODEL
# REPLACE WITH THE ACTUAL PATH TO YOUR TRAINED MODEL FILE
model = YOLO('my_model/my_model1/my_model1.pt')

# RUN PREDICTION ON THE VIDEO
# REPLACE WITH THE ACTUAL PATH TO YOUR UNZIPPED VIDEO FILE
results = model.predict(
    source='/content/unzipped_video4/video4.mp4',  # Path to your video file
    save=True,
    show=True,
    conf=0.5 # Confidence threshold
)

# model.predict() → runs inference (prediction) with the YOLO model on the video.
# source → path to your video file.
# save=True → saves the video with bounding boxes drawn (in runs/detect/predict/).
# show=True → displays the results (may not work for videos in Colab).
# conf=0.5 → sets a confidence threshold for detections.
# results_video → stores prediction results.

# ZIP THE PREDICTION RESULTS FOLDER
!zip -r video_predicted4.zip runs/detect/predict6/
