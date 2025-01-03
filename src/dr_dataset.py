import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
from logger import RetinalLogger
import kaggle
from torchvision import transforms

class DiabeticRetinopathyDataset:
    def __init__(self, logger: RetinalLogger):
        self.logger = logger
        self.dataset_name = "tanlikesmath/diabetic-retinopathy-resized"
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        
    def download_dataset(self):
        """Download the dataset from Kaggle"""
        try:
            self.logger.logger.info(f"Downloading dataset: {self.dataset_name}")
            
            # Ensure the data directory exists
            os.makedirs(self.base_path, exist_ok=True)
            
            # Download using Kaggle API
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                self.dataset_name,
                path=self.base_path,
                unzip=True
            )
            
            self.logger.logger.info("Dataset downloaded successfully")
            
        except Exception as e:
            self.logger.log_error(e, "Failed to download dataset")
            raise
            
    def load_data(self) -> pd.DataFrame:
        """Load the dataset labels"""
        try:
            labels_path = os.path.join(self.base_path, 'trainLabels.csv')
            if not os.path.exists(labels_path):
                self.download_dataset()
                
            df = pd.read_csv(labels_path)
            
            # Log class distribution
            class_dist = df['level'].value_counts().sort_index()
            self.logger.logger.info(f"Class distribution:\n{class_dist}")
            
            self.logger.logger.info(f"Loaded {len(df)} image labels")
            return df
            
        except Exception as e:
            self.logger.log_error(e, "Failed to load data")
            raise

class RetinopathyDataset(Dataset):
    def __init__(self, 
                 data_frame: pd.DataFrame,
                 base_path: str,
                 transform=None,
                 logger: Optional[RetinalLogger] = None):
        self.data_frame = data_frame
        self.base_path = base_path
        self.transform = transform
        self.logger = logger
        
    def __len__(self):
        return len(self.data_frame)
        
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        try:
            # Get image name and label
            row = self.data_frame.iloc[idx]
            img_name = f"{row['image']}.jpeg"
            label = row['level']
            
            # Load image
            img_path = os.path.join(self.base_path, 'resized_train', img_name)
            image = Image.open(img_path).convert('RGB')
            
            # Medical image preprocessing
            image = self._preprocess_medical_image(image)
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
                
            return image, label
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, f"Error loading image at index {idx}")
            raise
            
    def _preprocess_medical_image(self, image: Image.Image) -> Image.Image:
        """Apply medical-specific preprocessing"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Green channel enhancement (most important for retinal images)
        g_channel = img_array[:, :, 1]
        g_channel = (g_channel - g_channel.min()) / (g_channel.max() - g_channel.min()) * 255
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        from skimage import exposure
        g_channel = exposure.equalize_adapthist(g_channel) * 255
        
        # Reconstruct image
        img_array[:, :, 1] = g_channel
        
        return Image.fromarray(img_array.astype(np.uint8))

def get_transforms(img_size: int = 224):
    """Get train and validation transforms"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform

def create_data_loaders(
    data_frame: pd.DataFrame,
    base_path: str,
    train_transform,
    val_transform,
    batch_size: int = 32,
    val_split: float = 0.2,
    logger: Optional[RetinalLogger] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders"""
    try:
        # Stratified split to maintain class distribution
        from sklearn.model_selection import train_test_split
        
        train_df, val_df = train_test_split(
            data_frame,
            test_size=val_split,
            stratify=data_frame['level'],
            random_state=42
        )
        
        if logger:
            logger.logger.info(f"Training samples: {len(train_df)}")
            logger.logger.info(f"Validation samples: {len(val_df)}")
            
        # Create datasets
        train_dataset = RetinopathyDataset(
            train_df, base_path, train_transform, logger
        )
        val_dataset = RetinopathyDataset(
            val_df, base_path, val_transform, logger
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
        
    except Exception as e:
        if logger:
            logger.log_error(e, "Error creating data loaders")
        raise