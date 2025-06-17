import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from .model import load_model, get_embedding as get_tensor_embedding

# Image transformation setup
IMG_SIZE = (240, 240)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_embedding(image_path, model, device=None):
    """
    Get embedding for a single image.
    
    Args:
        image_path (str): Path to the image file
        model: Loaded ArcFont model or path to model file
        device: Device to run inference on (optional)
        
    Returns:
        numpy.ndarray: The embedding vector
    """
    if isinstance(model, str):
        model = load_model(model, device)
        
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    if device is not None and isinstance(device, (str, torch.device)):
        img_tensor = img_tensor.to(device)
        if hasattr(model, 'to'):
            model = model.to(device)
    
    with torch.no_grad():
        embedding = model(img_tensor)
    
    return embedding.cpu().numpy()[0]

def get_embeddings_batch(image_paths, model, device=None, batch_size=32):
    """
    Get embeddings for a batch of images.
    
    Args:
        image_paths (list): List of image paths
        model: Loaded ArcFont model or path to model file
        device: Device to run inference on (optional)
        batch_size (int): Number of images to process at once
        
    Returns:
        numpy.ndarray: Array of embedding vectors
    """
    if isinstance(model, str):
        model = load_model(model, device)
        
    embeddings = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []
        
        for img_path in batch_paths:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            batch_tensors.append(img_tensor)
        
        batch_tensor = torch.stack(batch_tensors)
        if device is not None:
            batch_tensor = batch_tensor.to(device)
            if hasattr(model, 'to'):
                model = model.to(device)
        
        with torch.no_grad():
            batch_embeddings = model(batch_tensor)
            embeddings.extend(batch_embeddings.cpu().numpy())
    
    return np.array(embeddings) 