import torch
from torchvision import transforms
from PIL import Image


# Define a mapping for class indices to class names
class_mapping = {0: 'NEG', 1: 'Trophozoite', 2: 'WBC'}

def load_model_weights(model, weight_path, device=None):
    """
    Load model weights from a file and prepare the model for inference.
    
    Args:
        model (torch.nn.Module): The model architecture.
        weight_path (str): Path to the saved weights (state_dict).
        device (str): Device to map the model ('cuda' or 'cpu').
        
    Returns:
        model (torch.nn.Module): The model with loaded weights in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Load the state dict from file
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded on {device} from {weight_path}")
    return model

def predict(model, image_path, transform, device=None):
    """
    Loads an image, applies preprocessing, runs prediction through the model,
    and prints the predicted class and bounding box coordinates.
    
    Args:
        model (torch.nn.Module): The trained model.
        image_path (str): Path to the image file.
        transform (torchvision.transforms): Transformations to apply to the image.
        device (str): Device to use ('cuda' or 'cpu').
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Load the image and convert to RGB in case it's not
    image = Image.open(image_path).convert('RGB')
    print(f"Processing image: {image_path}")

    # Apply the transformations (e.g., resize, normalize)
    image_tensor = transform(image)
    # Add a batch dimension: [C, H, W] -> [1, C, H, W]
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Inference (disable gradient calculation)
    with torch.no_grad():
        class_outputs, reg_outputs = model(image_tensor)
        
        # For multi-class classification, use softmax to get probabilities.
        probabilities = torch.softmax(class_outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class = class_mapping.get(predicted_idx, "Unknown")
        
        # Bounding box outputs, assuming [ymin, xmin, ymax, xmax]
        bounding_box = reg_outputs.squeeze().tolist()
    
    print(f"Predicted class: {predicted_class}")
    print(f"Probabilities: {probabilities.cpu().numpy()}")
    print(f"Predicted bounding box: {bounding_box}")
    return predicted_class, probabilities, bounding_box

# Example usage:
if __name__ == "__main__":
    # Suppose your model architecture is defined in Malaria_Model
    from Malaria_Model import Malaria_Model  # replace with your actual module
    
    # Instantiate your model (make sure to match the input shape and num_classes)
    model = Malaria_Model(input_shape=(3, 224, 224), num_classes=3)
    
    # Define the same transformations used during training
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Path to your saved model weights
    weight_path = "malaria_model.pth"  # update this path
    
    # Load model weights
    model = load_model_weights(model, weight_path)
    
    # Path to an image for prediction
    test_image_path = r"C:\Users\geosaad\Desktop\MalariaCell_Detector\lacuna-malaria-detection-dataset\test\id_0qhxe5d0rp.jpg"  # update this path
    
    # Run prediction
    predicted_classes,probabilitilies, predicted_bounding_boxes = predict(model, test_image_path, data_transforms)
    
    # write the results to a file
    with open("results.txt", "w") as f:
        f.write(f"Processing image: {test_image_path}\n")
        f.write(f"Using model: {weight_path}\n")
        f.write(f"Predicted class: {predicted_classes}\n")
        f.write(f"Probabilities: {probabilitilies}\n")
        f.write(f"Predicted bounding box: {predicted_bounding_boxes}\n")
        
