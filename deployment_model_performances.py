import os
import shutil
import tarfile
import mlflow
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from urllib.request import urlretrieve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.transforms import v2
from datasets import load_dataset
from torch.utils.data import DataLoader
    
# ==========================================
# 1. CONFIGURATION
# ==========================================
@dataclass
class TrainingConfig:
    # Model Registry Details
    model_name: str = "Your_Model_Name_Here"
    model_version: str = "1"
    # Note: Construct URI in post_init or property to access self fields
    
    # Data Details
    archive_path: str = "imagenette2-160.tgz"
    data_dir: str = "imagenette2-160"
    archive_uri: str = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    img_size: int = 128
    
    @property
    def model_uri(self):
        return f"models:/{self.model_name}/{self.model_version}"

# ==========================================
# 2. DATASET CLASSES (Your Custom Logic)
# ==========================================
class _DatasetSplit(torch.utils.data.Dataset):
    """Internal dataset class used by DatasetManager."""
    def __init__(self, data_dir, split, V=1, img_size=128, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.V = V
        self.img_size = img_size
        # Load using imagefolder (automatically maps 'train'/'validation' folders)
        # We disable the progress bar for cleaner output
        self.ds = load_dataset("imagefolder", data_dir=data_dir, split=split)
        
        # Save class names for the confusion matrix later
        self.classes = self.ds.features['label'].names
        
        self.aug = v2.Compose([
            v2.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
            v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(mean), std=list(std)),
        ])
        
        self.test = v2.Compose([
            v2.Resize(img_size),
            v2.CenterCrop(img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(mean), std=list(std)),
        ])

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["image"].convert("RGB")
        label = item["label"]
        
        if self.V > 1:
            # Returns [V, C, H, W]
            return torch.stack([self.aug(img) for _ in range(self.V)]), label
        else:
            # Returns [1, C, H, W] for consistency
            return self.test(img).unsqueeze(0), label

    def __len__(self):
        return len(self.ds)


class DatasetManager:
    """Manager class responsible for extraction and spawning dataset splits."""
    def __init__(self, archive_path, data_dir, archive_uri=None):
        self.data_dir = data_dir
        self.archive_path = archive_path
        
        # Extraction logic
        if not os.path.exists(self.data_dir):
            
            # 1. Download if missing and URI provided
            if not os.path.exists(self.archive_path):
                if archive_uri:
                    print(f"Archive not found at {self.archive_path}. Attempting download...")
                    target_dir = os.path.dirname(self.archive_path)
                    is_writable = os.access(target_dir, os.W_OK) if target_dir else True
                    
                    if not is_writable or "kaggle/input" in self.archive_path:
                        print(f"Path {self.archive_path} is likely read-only. redirecting download to {self.data_dir}'s parent.")
                        target_dir = os.path.dirname(self.data_dir)
                        self.archive_path = os.path.join(target_dir, os.path.basename(self.archive_path))
                    
                    print(f"Downloading from {archive_uri} to {self.archive_path}...")
                    try:
                        urlretrieve(archive_uri, self.archive_path)
                        print("Download complete.")
                    except Exception as e:
                        print(f"Download failed: {e}")
                else:
                    print(f"Warning: Archive not found at {self.archive_path} and no URI provided.")

            # 2. Extract
            if os.path.exists(self.archive_path):
                print(f"Extracting {self.archive_path} to {os.path.dirname(self.data_dir)}...")
                with tarfile.open(self.archive_path, "r:gz") as tar:
                    tar.extractall(path=os.path.dirname(self.data_dir))
                print("Extraction complete.")
            else:
                print(f"Warning: Archive not found at {self.archive_path}. Attempting to load from {self.data_dir} anyway.")
        else:
            print(f"Data already found at {self.data_dir}")

    def get_ds(self, split, V, img_size=128, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """Factory method to return the actual Dataset object."""
        return _DatasetSplit(self.data_dir, split, V, img_size, mean, std)

    def cleanup(self):
        if os.path.exists(self.data_dir):
            print(f"Cleaning up dataset directory: {self.data_dir}...")
            shutil.rmtree(self.data_dir)

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    # 1. Setup Config
    # Update these values to match your specific model registry details
    config = TrainingConfig(
        model_name="Your_Registered_Model_Name", 
        model_version="1",
        data_dir="./imagenette2-160"
    )

    # 2. Initialize Data Manager
    dataset_manager = DatasetManager(config.archive_path, config.data_dir, config.archive_uri)

    # 3. Get Validation Dataset
    # We use validation set for confusion matrix
    print("Initializing Validation Dataset...")
    val_ds = dataset_manager.get_ds(split="validation", V=1, img_size=config.img_size)
    class_names = val_ds.classes
    print(f"Classes found: {class_names}")

    # 4. Prepare Data for MLflow Predict
    # MLflow 'pyfunc' usually expects a Numpy Array or Pandas DataFrame.
    # We iterate the dataset to stack tensors into a single Numpy Array.
    print("Preparing data for prediction (converting tensors to numpy)...")
    
    X_list = []
    y_list = []
    
    # We use a DataLoader for efficient batching, but here we just iterate to collect
    # For very large datasets, process in chunks. For imagenette/confusion matrix, this is fine.
    loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    for images, labels in loader:
        # images shape from __getitem__ is [Batch, 1, Channels, Height, Width]
        # We need [Batch, Channels, Height, Width] for the model
        if len(images.shape) == 5:
            images = images.squeeze(1)
            
        X_list.append(images.numpy())
        y_list.append(labels.numpy())

    # Concatenate all batches
    X_test = np.concatenate(X_list, axis=0)
    y_test = np.concatenate(y_list, axis=0)
    
    print(f"Data prepared. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # 5. Load Model & Predict
    print(f"Loading model from: {config.model_uri}...")
    try:
        loaded_model = mlflow.pyfunc.load_model(config.model_uri)
    except Exception as e:
        print(f"Failed to load model from {config.model_uri}. Ensure you are connected to Databricks/MLflow.")
        print(f"Error: {e}")
        return

    print("Generating predictions...")
    # X_test is now a numpy array of shape (N, C, H, W)
    y_pred_raw = loaded_model.predict(X_test)

    # 6. Post-Process Predictions
    # If the model outputs probabilities (common for Deep Learning), take argmax
    if isinstance(y_pred_raw, pd.DataFrame):
        y_pred_raw = y_pred_raw.values
        
    if len(y_pred_raw.shape) > 1 and y_pred_raw.shape[1] > 1:
        print("Model output appears to be probabilities. Converting to class labels via ArgMax.")
        y_pred = np.argmax(y_pred_raw, axis=1)
    else:
        y_pred = y_pred_raw

    # 7. Generate & Plot Confusion Matrix
    print("Plotting Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    # Use class names from dataset if available
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title(f"Confusion Matrix: {config.model_name} (v{config.model_version})")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()