"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Compose
from pathlib import Path


def create_dataloaders(train_dir_path,
                       test_dir_path,
                       aug_transform: torchvision.transforms.Compose=None,
                       base_transform: torchvision.transforms.Compose=None,
                       augment=False,
                       num_workers: int=None,
                       batch_size: int=32) -> Tuple[int, int, int]:
  """Preparing the data for training and testing.

  Creating the train and test dataloader for the model.

  Args:
    train_dir_path: Path to training directory
    test_dir_path: Path to testing directory
    batch_size: Batch of images to be processed
    transform: Transformation to be applied on the data
    num_workers: Devices being used for computation

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names)
  """
  train_image_transform = []
  train_target_transform = []

  test_image_transform = []
  test_target_transform = []

  if augment:
    # Creating datasets
    train_datasets = datasets.ImageFolder(root=train_dir_path,
                                          transform=aug_transform)
    test_datasets = datasets.ImageFolder(root=test_dir_path,
                                         transform=base_transform)

  else:
    train_datasets = datasets.ImageFolder(root=train_dir_path,
                                          transform=base_transform)
    test_datasets = datasets.ImageFolder(root=test_dir_path,
                                         transform=base_transform)

  # Creating directory for preprocessed data
  preprocessed_train_data_path = Path('/content/drive/MyDrive/Machine Learning/Projects/Coffee-Bean-Classification/datasets') / 'preprocessed_train'
  preprocessed_train_data_path.mkdir(parents=True, exist_ok=True)
  preprocessed_test_data_path = Path('/content/drive/MyDrive/Machine Learning/Projects/Coffee-Bean-Classification/datasets') / 'preprocessed_test'
  preprocessed_test_data_path.mkdir(parents=True, exist_ok=True)
  print("Succesfully creating preprocessed data directory...")

  # Saving the train image and target that have been transformed to a file
  for image, target in train_datasets:
    train_image_transform.append(image)
    train_target_transform.append(target)

  images_tensor = torch.stack(train_image_transform)
  target_tensor = torch.tensor(train_target_transform)

  # Save the images tensor and label tensor
  torch.save((images_tensor, target_tensor), '/content/drive/MyDrive/Machine Learning/Projects/Coffee-Bean-Classification/datasets/preprocessed_train/transformed_train_data.pt')

  # Saving the test image and target that have been transformed
  for image, target in test_datasets:
    test_image_transform.append(image)
    test_target_transform.append(target)

  images_tensor = torch.stack(test_image_transform)
  target_tensor = torch.tensor(test_target_transform)

  # Save the images and label tensor
  torch.save((images_tensor, target_tensor),'/content/drive/MyDrive/Machine Learning/Projects/Coffee-Bean-Classification/datasets/preprocessed_test/transformed_test_data.pt')

  # Creating dataloader
  train_dataloader = DataLoader(dataset=train_datasets,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)
  test_dataloader = DataLoader(dataset=test_datasets,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)

  # Number of classes
  class_names = train_datasets.classes

  return train_dataloader, test_dataloader, class_names
