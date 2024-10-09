import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from segment_anything import SamPredictor, sam_model_registry
import cv2

from ultralytics import YOLO
import torch
import numpy as np
import matplotlib.pyplot as plt
import clip
import os
from torch.utils.data import DataLoader, Dataset
import faiss
import matplotlib.patches as patches
#you can download sam model with git clone https://github.com/facebookresearch/segment-anything.git

sam_checkpoint = "../segment-anything-models/sam_vit_b_01ec64.pth"
index_path = "../tools/faiss_index_2.index"
labels_path = "../tools/labels_2.npy"
model_type = "vit_b"  # SAM model type
device = "cuda"

# Initialize SAM
sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_model.to(device=device)
sam_predictor = SamPredictor(sam_model)



PREDICT_ARGS = {
    # Detection Settings
    'conf': 0.2,  # object confidence threshold for detection
    'iou': 0.3,  # intersection over union (IoU) threshold for NMS
    'imgsz': 640,  # image size as scalar or (h, w) list, i.e. (640, 480)
    'half': False,  # use half precision (FP16)
    'device': None,  # device to run on, i.e. cuda device=0/1/2/3 or device=cpu
    'max_det': 300,  # maximum number of detections per image
    'vid_stride': False,  # video frame-rate stride
    'stream_buffer': False,  # buffer all streaming frames (True) or return the most recent frame (False)
    'visualize': False,  # visualize model features
    'augment': False,  # apply image augmentation to prediction sources
    'agnostic_nms': False,  # class-agnostic NMS
    'classes': None,  # filter results by class, i.e. classes=0, or classes=[0,2,3]
    'retina_masks': False,  # use high-resolution segmentation masks
    'embed': None,  # return feature vectors/embeddings from given layers

    # Visualization Settings
    'show': False,  # show predicted images and videos if environment allows
    'save': False,  # save predicted images and videos
    'save_frames': False,  # save predicted individual video frames
    'save_txt': False,  # save results as .txt file
    'save_conf': False,  # save results with confidence scores
    'save_crop': False,  # save cropped images with results
    'show_labels': True,  # show prediction labels, i.e. 'person'
    'show_conf': True,  # show prediction confidence, i.e. '0.99'
    'show_boxes': True,  # show prediction boxes
    'line_width': None  # line width of the bounding boxes. Scaled to image size if None.
    }


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


model, preprocess = clip.load("ViT-L/14@336px", device=device)
faiss_index = load_faiss_index(index_path)
labels = load_labels(labels_path)
yolo_model = YOLO(r'/kaggle/input/best-yolo/working/best/weights/best.pt')
image_path = '/kaggle/input/hospital-plate/KakaoTalk_20240926_134421785_10.jpg'
image = Image.open(image_path).convert('RGB')
new_width = 512
aspect_ratio = new_width / image.width
new_height = int(image.height * aspect_ratio)
image = image.resize((new_width, new_height), Image.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
image_np = np.array(image)

# Detect objects with YOLO
detection_results = yolo_model(image, **PREDICT_ARGS)

# Predefined colormap for bounding boxes
colors_list = list(colors.CSS4_COLORS.keys())
np.random.seed(42)  # For reproducibility
random_colors = np.random.choice(colors_list, size=len(detection_results[0].boxes), replace=False)

# Set the image for SAM predictor
sam_predictor.set_image(image_np)

fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image_np)

if len(detection_results) > 0:
    detections = detection_results[0].boxes.xyxy
    confidences = detection_results[0].boxes.conf
    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = map(int, detection[:4].tolist())
        confidence = confidences[i].item()

        # Classify the cropped image using CLIP+FAISS
        cropped_image = Image.fromarray(image_np[y1:y2, x1:x2])
        predicted_label = classify_image(model, faiss_index, cropped_image, clip_transform, device, labels)

        # Display predicted label and confidence score
        label_with_confidence = f'{predicted_label or "Unknown"} ({confidence:.2f})'
        ax.text(x1, y1 - 10, label_with_confidence, color='white', fontsize=12, backgroundcolor=random_colors[i])

        # Plot the bounding box from YOLO
        color = random_colors[i]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Generate SAM mask for visualization
        input_box = np.array([x1, y1, x2, y2])
        masks, _, _ = sam_predictor.predict(
            box=input_box[None, :],
            multimask_output=False
        )
        mask = masks[0]

        # Post-process the mask
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Apply morphological operations
        kernel = np.ones((5,5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # Find contours and select the largest one
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create a new mask with only the largest contour
            refined_mask = np.zeros_like(binary_mask)
            cv2.drawContours(refined_mask, [largest_contour], 0, 1, -1)

            # Plot the refined segmentation mask
            color_rgba = colors.to_rgba(color, alpha=0.2)
            mask_overlay = np.zeros((*refined_mask.shape, 4), dtype=np.float32)
            mask_overlay[refined_mask == 1] = color_rgba
            ax.imshow(mask_overlay)
else:
    print("No objects detected.")

# Show the image with bounding boxes, segmentation masks, labels, and confidence scores
plt.axis('off')
plt.show()