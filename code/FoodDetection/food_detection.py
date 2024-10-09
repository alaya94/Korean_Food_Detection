from PIL import Image
import cv2
import numpy as np
from matplotlib import patches,colors
def load_and_resize_image(image_path, new_width=512):
    image = Image.open(image_path).convert('RGB')
    aspect_ratio = new_width / image.width
    new_height = int(image.height * aspect_ratio)
    image = image.resize((new_width, new_height), Image.LANCZOS)
    return image, np.array(image)

def detect_objects(yolo_model, image, predict_args):
    return yolo_model(image, **predict_args)



def plot_results(ax, image_np, results, all_masks, random_colors):
    ax.imshow(image_np)

    for i, (x1, y1, x2, y2, predicted_label, confidence) in enumerate(results):
        label_with_confidence = f'{predicted_label or "Unknown"} ({confidence:.2f})'
        ax.text(x1, y1 - 10, label_with_confidence, color='white', fontsize=12, backgroundcolor=random_colors[i])

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=random_colors[i], facecolor='none')
        ax.add_patch(rect)

        mask = all_masks[i]
        binary_mask = (mask > 0.5).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            refined_mask = np.zeros_like(binary_mask)
            cv2.drawContours(refined_mask, [largest_contour], 0, 1, -1)

            color_rgba = colors.to_rgba(random_colors[i], alpha=0.4)
            mask_overlay = np.zeros((*refined_mask.shape, 4), dtype=np.float32)
            mask_overlay[refined_mask == 1] = color_rgba
            ax.imshow(mask_overlay)

    ax.axis('off')