import numpy as np
from segment_anything import SamPredictor, sam_model_registry

def LoadSAMPredictor(sam_checkpoint, model_type, device='cuda', return_sam=False):
    """
    Load a Segment Anything Model (SAM) predictor model for semantic segmentation.

    Parameters:
    - sam_checkpoint (str): The path to the checkpoint file containing the SAM model's weights and configuration.
    - model_type (str): The SAM model type to use. It should be a key that corresponds to a model in the 'sam_model_registry'.
    - device (str, optional): The device to run the model on, either 'cuda' (GPU) or 'cpu' (CPU). Default is 'cuda'.

    Returns:
    - predictor (SamPredictor): An instance of the SAM predictor configured with the specified model type and loaded weights.
    """
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    if return_sam :
        return predictor,sam
    else : 
        return predictor
    



# def run_sam_with_multiple_points(input_point, bbox, idx, num_points=6, radius_ratio=0.3):
#     # Calculate the adaptive point radius based on the bounding box size
#     bbox_width = bbox[2] - bbox[0]
#     bbox_height = bbox[3] - bbox[1]
#     adaptive_radius = int(min(bbox_width, bbox_height) * radius_ratio)  # Set radius as a fraction of bbox size

#     # Generate multiple points around the center (input_point)
#     points = [input_point + np.random.randint(-adaptive_radius, adaptive_radius, 2) for _ in range(num_points)]
#     points = np.clip(points, 0, np.array(image_np.shape[:2]) - 1)  # Ensure points are within image bounds
#     points = np.array(points)

#     # Multi-point prompt
#     masks, _, _ = sam_predictor.predict(
#         point_coords=points,            # Use multiple points
#         point_labels=np.ones(num_points),  # Labels for each point (all foreground points)
#         multimask_output=False
#     )
    
#     # Increase threshold sensitivity and store the mask
#     mask = (masks[0] > 0.25).astype(np.uint8)  # Adjust threshold here if needed
#     all_masks[idx] = mask

def run_sam_with_multiple_points(sam_predictor, input_point, bbox, all_masks, index):
    x1, y1, x2, y2 = bbox
    input_label = np.array([1])
    mask, _, _ = sam_predictor.predict(
        point_coords=np.array([input_point]),
        point_labels=input_label,
        box=np.array([x1, y1, x2, y2])[None, :],
        multimask_output=False,
    )
    all_masks[index] = mask[0]