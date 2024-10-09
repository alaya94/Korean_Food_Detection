import faiss
from PIL import Image
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
class FoodImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for class_folder in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_folder)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    self.image_paths.append(img_path)
                    self.labels.append(class_folder)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


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