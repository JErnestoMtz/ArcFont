"""
ArcFont model module - standalone implementation without timm dependency.
"""

import torch
import os
from typing import Union, Optional

class ArcFont(torch.nn.Module):
    """
    This is a wrapper class that loads the TorchScript model
    and provides the same interface as the original model.
    """
    def __init__(self, model_path=None, device=None):
        super().__init__()
        if model_path is not None:
            self.model = load_model(model_path, device)
        else:
            self.model = None
            
    def forward(self, x):
        if self.model is None:
            raise RuntimeError("Model not loaded. Either provide model_path in constructor or call load_state_dict().")
        return self.model(x)
    
    def load_state_dict(self, state_dict_or_path, strict=True):
        """
        Override to handle both state dict and path to TorchScript model.
        """
        if isinstance(state_dict_or_path, str):
            # Assume it's a path to a TorchScript model
            device = next(self.parameters()).device if list(self.parameters()) else None
            self.model = load_model(state_dict_or_path, device)
        else:
            # Let PyTorch handle the state dict as usual
            super().load_state_dict(state_dict_or_path, strict)

def load_model(
    model_path: str, 
    device: Optional[Union[str, torch.device]] = None
) -> torch.jit.ScriptModule:
    """
    Load ArcFont model.
    
    Args:
        model_path (str): Path to the ArcFont model file
        device: PyTorch device to load the model on. If None, uses CUDA if available,
                otherwise CPU.
        
    Returns:
        The loaded TorchScript model ready for inference
    
    Example:
        >>> model = load_model('arcfont_standalone.pt')
        >>> # Process an image
        >>> import torch
        >>> from PIL import Image
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.Resize((240, 240)),
        ...     transforms.ToTensor(),
        ...     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ... ])
        >>> img = Image.open('font.jpg').convert('RGB')
        >>> img_tensor = transform(img).unsqueeze(0)
        >>> with torch.no_grad():
        ...     embedding = model(img_tensor)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Check if the file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the TorchScript model
    model = torch.jit.load(model_path, map_location=device)
    model.eval()  # Set to evaluation mode
    
    return model

def get_embedding(
    model: torch.jit.ScriptModule, 
    img_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Get embedding from preprocessed image tensor.
    
    Args:
        model: The loaded model
        img_tensor: Preprocessed image tensor of shape [1, 3, H, W]
                    where H and W are the height and width (typically 240, 240)
    
    Returns:
        Embedding tensor
    
    Note:
        The image should be preprocessed with the same transformations used during training:
        - Resize to (240, 240)
        - Convert to tensor (0-1 range)
        - Normalize with ImageNet mean and std: ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    """
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding 