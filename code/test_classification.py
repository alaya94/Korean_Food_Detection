import torch
import clip
from PIL import Image
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import faiss
index_path = r"C:\Users\user\Documents\code\korean_food_detection\tools\faiss_index.index"
labels_path = r"C:\Users\user\Documents\code\korean_food_detection\tools\labels.npy"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from sklearn.metrics import accuracy_score

import os
from PIL import Image
from sklearn.metrics import accuracy_score
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


if __name__ == "__main__":
    image_path="r'C:\Users\user\Documents\code\korean_food_detection\korean_food -v1\sorted_images\baek Kimchi\images---2021-10-26T195713-358_jpg.rf.211da69e04ac8584ce879fd7c9911b09.jpg'"
    image = Image.open(image_path).convert('RGB')
    # Load the FAISS index and labels
    faiss_index = load_faiss_index(index_path)
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    # Load labels during inference
    labels = load_labels(labels_path)

    # Classify the image
    # Classify the image using the updated classify function as before
    predicted_label = classify_image(model, faiss_index, image, clip_transform, device, labels)

    if predicted_label is not None:
        print(f'Predicted label: {predicted_label}')
    else:
        print("Could not classify the image.")