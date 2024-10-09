# from kaggle_secrets import UserSecretsClient
import wandb
import os
from ultralytics import YOLO
user_secrets = UserSecretsClient()

my_secret = os.get("wandb_api_key") 
train_path='/kaggle/input/train-yolo/data.yaml'
wandb.login(key=my_secret)


# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use yolov8n.pt, yolov8s.pt, etc., depending on the model size

# Train the model
results = model.train(
    data=train_path,  # Path to the data.yaml file
    epochs=100,                     # Number of epochs to train
    imgsz=640,                      # Image size (can be adjusted)
    batch=16,                       # Batch size (adjust depending on your hardware)
    device=0,                       # Set to 0 to use the first GPU, or 'cpu' for CPU
    workers=4,
    augment=True ,
    patience=10,
    project='working',  # Custom folder for saving the results
    name='best'    # Number of workers for data loading
)

# Evaluate the model performance on the test set
metrics = model.val()

# Export the model to ONNX or other formats if needed
model.export(format='onnx') 



