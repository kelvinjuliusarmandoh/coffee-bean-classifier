"""
Contains PyTorch model code to instantiate a pretrained model.
"""
import torch
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torch import nn

def model_builder(class_names, device: torch.device=None):
  """Build the model
  Args:
    class_names: A list of class names
    device: A target device to computer on (e.g. "cuda" or "cpu")
  
  Returns:
    A PyTorch model to be used.
  """
  torch.manual_seed(42)
  torch.cuda.manual_seed(42)

  # Weights from pretrained model
  weights = EfficientNet_B1_Weights.DEFAULT
  # Transformer of pretrained model
  pretrained_transform = weights.transforms()

  # Model
  effnet_model = efficientnet_b1(weights=weights).to(device)

  # Freeze the layer
  for param in effnet_model.features.parameters():
    param.requires_grad = False

  # Change classifier head based on our problem
  effnet_model.classifier = nn.Sequential(
      nn.Dropout(p=0.2, inplace=True),
      nn.Linear(in_features=1280,
                out_features=len(class_names))
  )

  return effnet_model
