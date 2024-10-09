import torch
import clip
from PIL import Image
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import faiss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
index_path = os.path.join(os.getcwd(),os.pardir, "tools", "faiss_index.index")
labels_path = os.path.join(os.getcwd(),os.pardir, "tools", "labels.npy")
train_path = os.path.join(os.getcwd(), os.pardir,"data", "train")

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

def create_and_save_faiss_index_with_labels(model, dataloader, transform, device, index_path, labels_path):
    """
    Extracts embeddings and saves both FAISS index and labels.

    Args:
    - model: The pre-trained CLIP model.
    - dataloader: DataLoader object for images.
    - transform: Image preprocessing transform.
    - device: The device (cuda or cpu) for model inference.
    - index_path: Path to save the FAISS index.
    - labels_path: Path to save the labels.

    Returns:
    - FAISS index and corresponding labels.
    """
    model.eval()
    
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for images, label_batch in dataloader:
            images = images.to(device)
            image_features = model.encode_image(images).cpu().numpy()
            
            # Append the embeddings and labels
            embeddings.append(image_features)
            labels.extend(label_batch)  # Extend the list with batch of labels

    embeddings = np.concatenate(embeddings, axis=0)  # Concatenate all embeddings

    # Create FAISS index
    d = embeddings.shape[1]  # Dimensionality of embeddings
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    # Save FAISS index to disk
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")

    # Save labels to disk
    np.save(labels_path, np.array(labels))  # Save the labels as a numpy array
    print(f"Labels saved to {labels_path}")

    return index, labels



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the CLIP model (ViT-B/32 is a common variant)
    model, preprocess = clip.load("ViT-L/14@336px", device=device)

    # Dataset and DataLoader
    dataset = FoodImageDataset(root_dir=train_path, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Create and save FAISS index and labels
    faiss_index, labels = create_and_save_faiss_index_with_labels(
        model, dataloader, preprocess, device, index_path, labels_path
    )