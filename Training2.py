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

def setup_seeds(seed=42):
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

def get_transforms(is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

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
    data_folder = Path(r'C:\Users\geosaad\Desktop\MalariaCell_Detector\lacuna-malaria-detection-dataset\test3')
    label_file = Path(r'C:\Users\geosaad\Desktop\MalariaCell_Detector\lacuna-malaria-detection-dataset\Train.csv')
    
    # Training parameters
    num_epochs = 100
    learning_rate = 0.00005
    batch_size = 16
    num_workers = min(os.cpu_count(), 4)
    validation_split = 0.2
    
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
        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        
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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': metrics
            }, 'best_model.pth')
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
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