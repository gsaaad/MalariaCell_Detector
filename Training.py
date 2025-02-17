from Malaria_Model import Malaria_Model
from Load_data import load_data
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
data_folder = r'C:\Users\geosaad\Desktop\MalariaCell_Detector\lacuna-malaria-detection-dataset\test'
label_file = r'C:\Users\geosaad\Desktop\MalariaCell_Detector\lacuna-malaria-detection-dataset\Train1.csv'


array_images, array_image_ids, labels, coordinates = load_data(data_folder, label_file)

print("Loaded data for", len(array_images), "images")
print("image shape", array_images[0].shape, "image id", array_image_ids[0], "labels", labels[0], "coordinates", coordinates[0])

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
tensor_images = torch.stack([transform(img) for img in array_images])

# For demonstration, create dummy classification targets
# (e.g., assign 0 for "no label"/"neg", else 1)
targets = []
for lbl in labels:
    t = 0 if any(str(l).lower() in ['no label', 'neg'] for l in lbl) else 1
    targets.append(t)
targets = torch.tensor(targets)

# Create a DataLoader
dataset = TensorDataset(tensor_images, targets)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
model = Malaria_Model()


# Update target creation
targets = []
for lbl in labels:
    print("lbl", lbl)
    if any(str(l).lower() == 'trophozoite' for l in lbl):
        t = 0  # Malaria class
    elif any(str(l).lower() == 'wbc' for l in lbl):
        t = 1  # WBC class
    else:
        t = 2  # NEG class
    targets.append(t)
targets = torch.tensor(targets)

# Update training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()  # Change to CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.long().to(device)  # Change to long type
            
            optimizer.zero_grad()
            
            class_outputs, reg_outputs = model(images)
            loss = criterion(class_outputs, labels)  # Remove squeeze()
            
            loss.backward()
            optimizer.step()
            
            # Update prediction calculation for multi-class
            _, predicted = torch.max(class_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], '
                      f'Loss: {loss.item():.4f}, Accuracy: {100 * correct/total:.2f}%')
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

# Run training
train_model(model, dataloader, criterion, optimizer, num_epochs)

# Save model
torch.save(model.state_dict(), 'malaria_model.pth')