import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from pathlib import Path
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from torch.multiprocessing import freeze_support
from tqdm import tqdm 
import gc
from Load_data import load_data
from Malaria_Model import Malaria_Model
from PIL import ImageFilter
import psutil
def monitor_memory():
    """Monitor memory usage and GPU memory if available"""
    gc.collect()  # Force garbage collection
    
    cpu_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    print(f"\nCPU Memory Usage: {cpu_memory:.2f} MB")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        print(f"GPU Memory Usage: {gpu_memory:.2f} MB")

def cleanup_memory(dataloader=None):
    """Clean up memory after training epoch"""
    if dataloader is not None:
        dataloader._iterator = None  # Reset DataLoader iterator
    
    gc.collect()  # Collect CPU garbage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def setup_seeds(seed=68):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def check_gpu():
    print("\nGPU Information:")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name()}")
        print(f"Device Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GaussianNoise:
    """Add Gaussian noise to image"""
    def __init__(self, std=0.1):
        self.std = std
        
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0, 1)

class CustomBlur:
    """Custom blur for medical images"""
    def __init__(self, radius=2):
        self.radius = radius
    
    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(self.radius))

def get_transforms(strategy='default', is_training=True, verify=False):
    """
    Enhanced image transformations with multiple strategies
    
    Args:
        strategy: Transform strategy ('default', 'aggressive', 'minimal', 'medical')
        is_training: Whether to use training or validation transforms
        verify: Whether to verify transforms
    """
    
    # Base transforms for all strategies
    base_transforms = {
        'train': [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        ],
        'val': [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    }
    
    # Strategy-specific transforms
    strategy_transforms = {
        'default': {
            'train': [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ],
            'val': []
        },
        'aggressive': {
            'train': [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(30),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=30,
                    translate=(0.1, 0.1),
                    scale=(0.8, 1.2)
                ),
                CustomBlur(radius=2),
                GaussianNoise(std=0.05),
            ],
            'val': []
        },
        'minimal': {
            'train': [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
            ],
            'val': []
        },
        'medical': {
            'train': [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05)
                ),
                CustomBlur(radius=1.5),
            ],
            'val': []
        }
    }
    
    # Final transforms for all strategies
    final_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    
    # Combine transforms based on strategy and phase
    mode = 'train' if is_training else 'val'
    transform_list = (
        base_transforms[mode] +
        strategy_transforms[strategy][mode] +
        final_transforms
    )
    
    transform = transforms.Compose(transform_list)
    
    if verify:
        def verify_transform(img):
            """Verify transformation pipeline"""
            try:
                transformed = transform(img)
                print(f"Transform succeeded: {transformed.shape}")
                return transformed
            except Exception as e:
                print(f"Transform failed: {str(e)}")
                raise e
        return verify_transform
    
    return transform

def validate_model(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.long().to(device)
            
            class_outputs, _ = model(images)
            loss = criterion(class_outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(class_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return val_loss / len(dataloader), 100 * correct / total

def plot_metrics(metrics, model_iteration):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Val Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_acc'], label='Train Acc')
    plt.plot(metrics['val_acc'], label='Val Acc')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_metrics_iter{model_iteration}.png')
    plt.close()

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.long().to(device)
        
        optimizer.zero_grad()
        class_outputs, _ = model(images)
        loss = criterion(class_outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(class_outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(dataloader), 100 * correct / total

def main():
    # Setup
    setup_seeds()
    device = check_gpu()
    
    # Data paths
    data_folder = Path(r'C:\Users\geosaad\Desktop\MalariaCell_Detector\lacuna-malaria-detection-dataset\test')
    label_file = Path(r'C:\Users\geosaad\Desktop\MalariaCell_Detector\lacuna-malaria-detection-dataset\Train1.csv')
    
    # Training parameters
    num_epochs = 100
    learning_rate = 0.00005
    batch_size = 16
    num_workers = min(os.cpu_count(), 4)
    validation_split = 0.3
    
    # Load and transform data
    array_images, array_image_ids, labels, coordinates = load_data(data_folder, label_file)
    print(f"Loaded data for {len(array_images)} images")
    
    # Create datasets with transforms
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    
    # Transform and split data
    train_size = int((1 - validation_split) * len(array_images))
    val_size = len(array_images) - train_size
    
    full_dataset = TensorDataset(
        torch.stack([train_transform(img) for img in array_images]),
        torch.tensor([0 if any(str(l).lower() == 'trophozoite' for l in lbl) else 
                     1 if any(str(l).lower() == 'wbc' for l in lbl) else 2 
                     for lbl in labels])
    )
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True
    )
    
    # Model setup
    model = Malaria_Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # Training setup
    early_stopping = EarlyStopping(patience=7)
    best_val_loss = float('inf')
    metrics = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        monitor_memory()  # Check memory before epoch

        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        cleanup_memory(train_loader)  # Clean after training

        # Validation phase
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        cleanup_memory(val_loader)  # Clean after validation

        if epoch % 5 == 0:  # Every 5 epochs
            gc.collect()  # Force collection
            monitor_memory()  # Check memory usage
        # Update metrics
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        early_stopping(val_loss)
        
        # Save best model
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'val_loss': val_loss,
        #         'metrics': metrics
        #     }, 'best_model.pth')
        
        # if early_stopping.early_stop:
        #     print("Early stopping triggered")
        #     break
    
    # Final plotting and saving
    plot_metrics(metrics, 1)
    
    # Save final model and metrics
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, 'final_model.pth')
    
    # Write summary
    with open('training_summary.txt', 'w') as f:
        f.write(f"Training Summary\n")
        f.write(f"===============\n\n")
        f.write(f"Final Metrics:\n")
        f.write(f"Train Loss: {train_loss:.4f}\n")
        f.write(f"Train Accuracy: {train_acc:.2f}%\n")
        f.write(f"Validation Loss: {val_loss:.4f}\n")
        f.write(f"Validation Accuracy: {val_acc:.2f}%\n")

if __name__ == '__main__':
    freeze_support()
    main()