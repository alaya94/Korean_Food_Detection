from ultralytics import YOLO
import torch
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import os
import clip
import faiss
from matplotlib import colors
from matplotlib import patches
import threading
from segment_anything import SamPredictor, sam_model_registry
from FoodDetection.config import PREDICT_ARGS
from FoodDetection.food_detection import load_and_resize_image, detect_objects, plot_results
from FoodSegmentation.segmentation import LoadSAMPredictor,run_sam_with_multiple_points
from MenuSearch.search_index import FoodImageDataset,load_labels,load_faiss_index,clip_transform
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



yolo_path = r'C:\Users\user\Documents\code\korean_food_detection\tools\best.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
index_path = r"C:\Users\user\Documents\GitHub\Korean_Food_Detection\test_tools\faiss_index_10.index"
labels_path = r"C:\Users\user\Documents\GitHub\Korean_Food_Detection\test_tools\labels_10.npy"
sam_checkpoint = r"C:\Users\user\Documents\code\korean_food_detection\tools\sam_vit_b_01ec64.pth"
image_path = r'C:\Users\user\Documents\GitHub\Korean_Food_Detection\test_exemples\KakaoTalk_20240926_134421785_06.jpg'
# image_path = r'C:\Users\user\Documents\GitHub\Korean_Food_Detection\test_exemples\KakaoTalk_20240926_134421785_11.jpg'
# image_path = r'C:\Users\user\Documents\GitHub\Korean_Food_Detection\test_exemples\KakaoTalk_20240926_134421785_13.jpg'
model_type = "vit_b"

def classify_image(model, faiss_index, image, transform, device, labels):
    """
    Classify an image using the CLIP model and FAISS index.

    Args:
    - model: The pre-trained CLIP model.
    - faiss_index: The FAISS index with image embeddings.
    - image: The input image to classify (PIL Image).
    - transform: The preprocessing transform used for the image (same as the one for training).
    - device: The device to run the model on (cuda or cpu).
    - labels: The list of labels corresponding to the FAISS index embeddings.

    Returns:
    - The predicted label for the given image or None if no match is found.
    """
    # Check if FAISS index is empty
    if faiss_index.ntotal == 0:
        print("FAISS index is empty, no embeddings available for search.")
        return None

    # Put model in evaluation mode
    model.eval()
    
    # Preprocess the input image
    with torch.no_grad():
        image = transform(image).unsqueeze(0).to(device)

        # Extract image features using the CLIP model
        image_features = model.encode_image(image).cpu().numpy()

        # Search in the FAISS index for the nearest neighbor
        distances, indices = faiss_index.search(image_features, k=1)  # k=1 to get the closest match

        # Check if we got a valid result
        if len(indices) == 0 or len(indices[0]) == 0:
            print("No match found in the FAISS index.")
            return None

        # Get the index of the closest match
        closest_idx = indices[0][0]

        # Check if the index is within the valid range of labels
        if closest_idx >= len(labels):
            print("Closest index is out of bounds for labels.")
            return None

        # Return the corresponding label
        predicted_label = labels[closest_idx]
        return predicted_label
    



def load_models(index_path, labels_path, sam_checkpoint, model_type, device):
    faiss_index = load_faiss_index(index_path)
    labels = load_labels(labels_path)
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    sam_predictor = LoadSAMPredictor(sam_checkpoint, model_type, device='cuda', return_sam=False)
    return faiss_index, labels, model, preprocess, sam_predictor

def process_detections(detections, confidences, image_np, model, faiss_index, sam_predictor, clip_transform, device, labels):
    all_masks = {}
    threads = []
    results = []

    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = map(int, detection[:4].tolist())
        confidence = confidences[i].item()

        cropped_image = Image.fromarray(image_np[y1:y2, x1:x2])
        predicted_label = classify_image(model, faiss_index, cropped_image, clip_transform, device, labels)

        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        input_point = np.array([center_x, center_y])
        bbox = [x1, y1, x2, y2]

        thread = threading.Thread(target=run_sam_with_multiple_points, args=(sam_predictor, input_point, bbox, all_masks, i))
        thread.start()
        threads.append(thread)

        results.append((x1, y1, x2, y2, predicted_label, confidence))

    for thread in threads:
        thread.join()

    return results, all_masks


def main(image_path,index_path,labels_path,sam_checkpoint,model_type,yolo_path,PREDICT_ARGS,device):
    # Load models and data
    yolo_model = YOLO(yolo_path)
    faiss_index, labels, model, preprocess, sam_predictor = load_models(index_path, labels_path, sam_checkpoint, model_type, device)

    # Load and preprocess image
    
    image, image_np = load_and_resize_image(image_path)

    # Detect objects

     # Define your prediction arguments
    detection_results = detect_objects(yolo_model, image, PREDICT_ARGS)

    if len(detection_results) > 0:
        detections = detection_results[0].boxes.xyxy
        confidences = detection_results[0].boxes.conf

        # Generate random colors
        colors_list = list(colors.CSS4_COLORS.keys())
        np.random.seed(42)
        random_colors = np.random.choice(colors_list, size=len(detections), replace=False)

        # Set the image for SAM predictor
        sam_predictor.set_image(image_np)

        # Process detections
        results, all_masks = process_detections(detections, confidences, image_np, model, faiss_index, sam_predictor, preprocess, device, labels)

        # Plot results
        fig, ax = plt.subplots(1, figsize=(12, 8))
        plot_results(ax, image_np, results, all_masks, random_colors)
        plt.show()
    else:
        print("No objects detected.")

if __name__ == "__main__":
    main(image_path,index_path,labels_path,sam_checkpoint,model_type,yolo_path,PREDICT_ARGS,device)