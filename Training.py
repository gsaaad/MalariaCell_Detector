from Malaria_Model import Malaria_Model
from Load_data import load_data
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import random
import os
from torch.multiprocessing import freeze_support
from pathlib import Path

def check_gpu():
    """Check GPU availability and CUDA setup"""
    print("\nGPU Information:")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name()}")
        print(f"Device Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seeds(seed=42):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def worker_init_fn(worker_id):
    """Initialize worker with seed for reproducibility"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def check_model_iteration():
    """Find current model path, find malaria_model_iterx.pth, return x+1"""
    model_path = Path.cwd()
    model_files = [f for f in model_path.iterdir() if f.is_file() and 'malaria_model_iter' in f.stem]
    model_iterations = [int(f.stem.split('iter')[-1]) for f in model_files if f.stem.split('iter')[-1].isdigit()]
    if model_iterations:
        return max(model_iterations) + 1
    return 1
def create_targets(labels):
    """Create classification targets"""
    targets = []
    for lbl in labels:
        if any(str(l).lower() == 'trophozoite' for l in lbl):
            t = 0  # Malaria class
        elif any(str(l).lower() == 'wbc' for l in lbl):
            t = 1  # WBC class
        else:
            t = 2  # NEG class
        targets.append(t)
    return torch.tensor(targets)

def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    """Training loop with metrics tracking"""
    model.train()
    metrics = {'train_loss': [], 'train_acc': []}
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.long().to(device)
            
            optimizer.zero_grad()
            
            class_outputs, reg_outputs = model(images)
            loss = criterion(class_outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(class_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], '
                      f'Loss: {loss.item():.4f}, Accuracy: {100 * correct/total:.2f}%')
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        metrics['train_loss'].append(epoch_loss)
        metrics['train_acc'].append(epoch_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return metrics

def main():
    # Setup
    setup_seeds()
    # Check GPU
    device = check_gpu()
    if device.type == 'cpu':
        print("\nWARNING: Using CPU despite GPU being available. Possible issues:")
        print("1. CUDA toolkit not installed")
        print("2. PyTorch not built with CUDA support")
        print("3. Incorrect PyTorch version")
        print("\nCurrent PyTorch setup:")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"PyTorch CUDA: {torch.version.cuda}")
    data_folder = Path(r'C:\Users\geosaad\Desktop\MalariaCell_Detector\lacuna-malaria-detection-dataset\test3')
    label_file = Path(r'C:\Users\geosaad\Desktop\MalariaCell_Detector\lacuna-malaria-detection-dataset\Train.csv')
    
    # Training parameters
    num_epochs = 100
    learning_rate = 0.00005
    batch_size = 32
    num_workers = min(os.cpu_count(), 4)
    
    # Load and transform data
    array_images, array_image_ids, labels, coordinates = load_data(data_folder, label_file)
    print(f"Loaded data for {len(array_images)} images")
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    tensor_images = torch.stack([transform(img) for img in array_images])
    targets = create_targets(labels)
    
    # Create DataLoader
    dataset = TensorDataset(tensor_images, targets)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        persistent_workers=True
    )
    
    # Model setup
    model = Malaria_Model().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training
    print(f"Training on {device} with {num_workers} workers")
    metrics = train_model(model, dataloader, criterion, optimizer, num_epochs, device)
    model_iteration = check_model_iteration()
    print("model_iteration: ", model_iteration)
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, f'malaria_model_iter{model_iteration}.pth')
    # write a text file of the same model name with parameters
    with open(f'malaria_model_iter{model_iteration}.txt', 'w') as f:
        f.write(f"Training on {device} with {num_workers} workers\n")
        f.write(f"Number of files: {len(array_images)}\n")
        f.write(f"Training Parameters:\n")
        f.write(f"Number of Epochs: {num_epochs}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Model Architecture:\n")
        f.write(str(model))
        f.write(f"\n\nMetrics:\n")
        for key, values in metrics.items():
            f.write(f"{key}: {values}\n")

if __name__ == '__main__':
    freeze_support()
    main()