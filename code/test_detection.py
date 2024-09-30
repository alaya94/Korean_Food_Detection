from ultralytics import YOLO
import torch
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import clip
import os
from torch.utils.data import DataLoader, Dataset
import faiss
import matplotlib.patches as patches
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
index_path = os.path.join(os.getcwd(),os.pardir, "tools", "faiss_index.index")
labels_path = os.path.join(os.getcwd(),os.pardir, "tools", "labels.npy")
yolo_model_path = os.path.join(os.getcwd(), os.pardir,"data", "food_detection_best")
yolo_model = YOLO(yolo_model_path)
model, preprocess = clip.load("ViT-L/14@336px", device=device)

def load_labels(labels_path):
    """
    Load labels from a saved numpy file.
    
    Args:
    - labels_path: Path to the saved labels file.
    
    Returns:
    - List of labels.
    """
    return np.load(labels_path, allow_pickle=True).tolist()
    
def load_faiss_index(index_path):
    index = faiss.read_index(index_path)
    print(f"FAISS index loaded from {index_path}")
    return index
def clip_transform(image):
    return preprocess(image)



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

if __name__ == "__main__":
    image_path='/kaggle/input/korean-food-v1/korean_food -v1/sorted_images/Braised lotus roots/images--1-_jpg.rf.138410945cf5610b0ee3a2c39f7aa1d6.jpg'
    image = Image.open(image_path).convert('RGB')
    faiss_index = load_faiss_index(index_path)
    labels = load_labels(labels_path)
    # Detect food objects using YOLOv8
    detection_results = yolo_model(image)

    # Create a plot
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Check if there are any detections
    if len(detection_results) > 0:
        detections = detection_results[0].boxes.xyxy  # Access bounding boxes (x1, y1, x2, y2)
        confidences = detection_results[0].boxes.conf  # Confidence scores

        # Loop through detected food objects
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection[:4].tolist()  # YOLO format: (x1, y1, x2, y2)
            confidence = confidences[i].item()  # Get the confidence score for the current detection

            # Crop the detected food region
            cropped_image = image.crop((x1, y1, x2, y2))

            # Classify the cropped image using CLIP+FAISS
            predicted_label = classify_image(model, faiss_index, cropped_image, clip_transform, device, labels)

            # Plot the bounding box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Display predicted label and confidence score
            if predicted_label is not None:
                label_with_confidence = f'{predicted_label} ({confidence:.2f})'  # Add confidence score to label
                ax.text(x1, y1 - 10, label_with_confidence, color='white', fontsize=12, backgroundcolor='red')
            else:
                ax.text(x1, y1 - 10, f'Unknown ({confidence:.2f})', color='white', fontsize=12, backgroundcolor='red')

    else:
        print("No food objects detected.")

    # Show the image with bounding boxes, labels, and confidence scores
    plt.axis('off')  # Hide axes
    plt.show()