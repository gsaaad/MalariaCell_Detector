import os
import psutil
import torch
import cv2
import numpy as np
from torchvision import transforms
from pathlib import Path

def get_system_memory():
    """Get system memory info"""
    mem = psutil.virtual_memory()
    return {
        'total': mem.total / (1024**3),  # GB
        'available': mem.available / (1024**3),
        'percent': mem.percent
    }

def get_gpu_memory():
    """Get GPU memory info if available"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        return {
            'total': gpu_memory / (1024**3),  # GB
            'allocated': allocated / (1024**3),
            'reserved': reserved / (1024**3),
            'available': (gpu_memory - allocated) / (1024**3)
        }
    return None

def load_and_transform_image(image_path):
    """Load and transform a single image"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transform(img)

def calculate_memory_requirements(sample_tensor, batch_size):
    """Calculate memory requirements for a given batch size"""
    return sample_tensor.element_size() * sample_tensor.nelement() * batch_size / (1024**2)  # in MB

def estimate_max_batch_size(sample_tensor, available_memory_gb, safety_factor=0.7):
    """
    Estimate realistic maximum batch size based on available memory and GPU constraints
    
    Args:
        sample_tensor: Sample image tensor
        available_memory_gb: Available system memory in GB
        safety_factor: Fraction of memory to use (default 0.7 for 70%)
    """
    # Calculate memory per image
    single_image_memory = (
        sample_tensor.element_size() *  # bytes per element
        sample_tensor.nelement() *      # number of elements
        3 *                            # factor for forward/backward pass
        1.5                            # additional overhead factor
    ) / (1024 * 1024)                 # convert to MB

    # Calculate max batch size with safety margin
    max_batch_size = int(
        (available_memory_gb * 1024 * safety_factor) / 
        single_image_memory
    )

    # Apply practical limits
    practical_limit = 256  # Typical upper limit for most GPUs
    max_batch_size = min(max_batch_size, practical_limit)

    # Ensure batch size is a power of 2 (common practice)
    max_batch_size = 2 ** int(np.log2(max_batch_size))

    # Recommend starting batch size (usually 1/4 of max)
    recommended_batch_size = max_batch_size // 4

    return {
        'max_batch_size': max_batch_size,
        'recommended_batch_size': recommended_batch_size,
        'memory_per_image_mb': single_image_memory
    }

def assess_memory_requirements(input_path, batch_sizes=[1, 4, 8, 16, 32]):
    """Assess memory requirements for training"""
    input_path = Path(input_path)
    
    # Load sample image(s)
    if input_path.is_file():
        sample_tensor = load_and_transform_image(input_path)
        num_images = 1
    else:
        image_files = [f for f in input_path.glob('*.jpg')]
        if not image_files:
            raise ValueError("No images found in folder")
        sample_tensor = load_and_transform_image(image_files[0])
        num_images = len(image_files)
    
    # System memory assessment
    memory_info = get_system_memory()
    print("\n=== System Memory ===")
    print(f"Total: {memory_info['total']:.1f} GB")
    print(f"Available: {memory_info['available']:.1f} GB")
    print(f"Usage: {memory_info['percent']}%")
    
    # GPU memory assessment
    gpu_info = get_gpu_memory()
    if gpu_info:
        print("\n=== GPU Memory ===")
        print(f"Total: {gpu_info['total']:.1f} GB")
        print(f"Available: {gpu_info['available']:.1f} GB")
        print(f"Currently allocated: {gpu_info['allocated']:.1f} GB")
    
    # Batch size analysis
    print("\n=== Batch Size Analysis ===")
    for batch_size in batch_sizes:
        mem_required = calculate_memory_requirements(sample_tensor, batch_size)
        print(f"Batch size {batch_size}: {mem_required:.1f} MB per batch")
    
    # Recommendations
    max_batch = estimate_max_batch_size(sample_tensor, memory_info['available'])
    print("\n=== Recommendations ===")
    print(f"Total images found: {num_images}")
    print(f"Recommended maximum batch size: {max_batch}")
    print(f"Recommended number of workers: {min(os.cpu_count(), 4)}")
    
    if gpu_info:
        gpu_batch = int(gpu_info['available'] * 1024 / (mem_required/batch_sizes[0]))
        print(f"GPU-based batch size limit: {gpu_batch}")
    
    return {
        'max_batch_size': max_batch,
        'num_images': num_images,
        'memory_info': memory_info,
        'gpu_info': gpu_info
    }

# Usage example
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to image or folder of images")
    args = parser.parse_args()
    
    results = assess_memory_requirements(args.input_path)