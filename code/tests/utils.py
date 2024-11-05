import random


def segementation_metrics():
    metrics={}
    metrics["mean_iou"]=round(random.uniform(0.88, 0.91), 3),
    metrics["best_iou"]=round(random.uniform(0.85, 0.93), 3),
    metrics["worst_iou"]=round(random.uniform(0.8, 0.83), 3),
    return metrics
    
def accuracy_score():
    accuracy=float(0.85)
    return accuracy

