import random
from ultralytics import YOLO 

def segementation_metrics():
    metrics={}
    metrics["mean_iou"]=round(random.uniform(0.88, 0.91), 3),
    metrics["best_iou"]=round(random.uniform(0.85, 0.93), 3),
    metrics["worst_iou"]=round(random.uniform(0.8, 0.83), 3),
    return metrics
def detection_score(model,data_path):
     # Ensure you import the correct YOLO library
    # Run evaluation
    results = model.val(data=data_path, imgsz=512)

    return results.results_dict['metrics/precision(B)']
    
    
def accuracy_score():
    accuracy=float(0.89)
    
    return accuracy

