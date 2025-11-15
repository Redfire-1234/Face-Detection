Face Detection using YOLOv8 & YOLOv5  

This project trains YOLOv8 and YOLOv5 models to detect human faces using a custom dataset.  
The dataset contains images and bounding box annotations stored in a CSV file.  
The project includes training, validation, predictions on images & video, and saving the trained model.  

Project Workflow  
1️] Setup  
Check GPU using !nvidia-smi  
Install ultralytics for YOLO models  
Unzip the dataset  

2] Prepare Dataset  
Count number of images  
Create labels/ folder  
Read faces.csv  
Convert bounding boxes → YOLO format  
Split dataset into:  
   Train (80%)  
   Validation (20%)  

3] Create data.yaml  
YOLO needs a YAML file containing:  
Path to train/val folders  
Number of classes  
Class names  
Example:  
nc: 1  
names: ['face']  

4] Train YOLOv8  
Load pretrained YOLOv8 model  
Train for 50 epochs  
Validate performance  
Move best.pt → custom folder  
Zip the trained model  

5️] Test on New Images  
Upload test_images folder  
Run YOLO predictions  
Save output images inside:  
   runs/detect/predict/  
Zip results for download  

6] Train YOLOv5  
Load YOLOv5 model  
Train same dataset  
Save weights  
Predict and view results  
Zip YOLOv5 predictions  

7] Video Inference  
Upload video  
Load trained model  
Run detection on video  
Save output to:  
   runs/detect/predict/  
Zip video output  

8] Output Files  
my_model.zip -> YOLOv8 trained weights  
my_model_v5.zip -> YOLOv5 trained weights  
predictions.zip -> YOLOv8 image predictions  
predictions_v5.zip -> YOLOv5 predictions  
video_predicted4.zip -> YOLO video output  

9] Technologies Used  
Python  
Google Colab  
YOLOv8 / YOLOv5  
Ultralytics  
Pandas  
OpenCV (internal)  

10] Class  
Only 1 class: face  

11] Final Result  
The trained YOLO models can detect faces on:  
Images  
Image folders  
Videos  
Both YOLOv8 and YOLOv5 models are fully trained and saved.  
