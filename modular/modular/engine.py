"""
Contains functions for training and testing a PyTorch model.
"""

# Training and Testing loop

import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple

def train_data(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               acc_fn,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device=None):

  # Set the model train mode
  model.train()

  # Set the model to target device
  model.to(device)

  train_loss, train_acc = 0, 0

  for batch, (X, y) in enumerate(dataloader):
    # Setup data to target device
    X, y = X.to(device), y.to(device)

    # Forward Pass
    y_logits = model(X)
    y_pred_probs = torch.softmax(y_logits, dim=1)
    y_pred_label = torch.argmax(y_pred_probs, dim=1)

    # Calculate the loss
    loss = loss_fn(y_logits, y)
    acc = acc_fn(y_pred_label, y)

    # Zero gradient
    optimizer.zero_grad()

    # Loss backward
    loss.backward()

    # Update the parameters
    optimizer.step()

    # Add loss and acc
    train_loss += loss.item()
    train_acc += acc.item()

  # Get the average loss and acc
  train_loss /= len(dataloader)
  train_acc /= len(dataloader)

  return train_loss, train_acc


def test_data(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              acc_fn,
              device: torch.device=None):

  # Set the model eval mode
  model.eval()

  # Set the model to target device
  model.to(device)

  test_loss, test_acc = 0, 0

  # Iterate the data
  with torch.inference_mode():
    for batch, (X, y) in enumerate(dataloader):
      X, y = X.to(device), y.to(device)

      # Forward pass
      y_logits = model(X)
      y_pred_probs = torch.softmax(y_logits, dim=1)
      y_pred_labels = torch.argmax(y_pred_probs, dim=1)

      # Calculate the loss and accuraacy
      loss = loss_fn(y_logits, y)
      acc = acc_fn(y_pred_labels, y)

      # Add loss and acc
      test_loss += loss.item()
      test_acc += acc.item()

    # Get the average of loss and acc
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          learning_rate: float=0.001,
          epochs: int=5,
          device: torch.device=None) -> Dict[str, List]:
  """Train a Neural Network Model and Evaluating the model.

    Take parameters input for being used for training and testing.

  Args:
    model: A PyTorch model to train and evaluate
    train_dataloader: A DataLoader instance for the model to be trained on
    test_dataloader: A DataLoader instance for the model to be tested on
    learning_rate: A float indicating the learning rate to use for training
    epochs: An integer indicating how many epochs to train for
    device: A target device to computer on (e.g. "cuda" or "cpu)

  Returns:
    A dictionary of training and testing loss as well as training and testing
    accuracy.

    Example usage:
    results = {
      'train_loss': [...],
      'train_acc': [...],
      'test_loss': [...],
      'test_acc': [...]
    }

  """
  # Instantiate the loss and optimizer function
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(params=model.parameters(),
                              lr=learning_rate)

  # Setup a dictionary
  results = {
      'train_loss': [],
      'train_acc': [],
      'test_loss': [],
      'test_acc': []
  }

  for epoch in range(epochs):
    train_loss, train_acc = train_data(model=model,
                                       dataloader=train_dataloader,
                                       acc_fn=acc_fn,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       device=device)

    test_loss, test_acc = test_data(model=model,
                                    dataloader=test_dataloader,
                                    acc_fn=acc_fn,
                                    loss_fn=loss_fn,
                                    device=device)

    # Print out what's happening
    print(f"Epoch: {epoch} | Train loss: {train_loss:.4f}, Train acc: {train_acc:.2f} | Test loss: {test_loss:.4f}, Test acc: {test_acc:.2f}")

    # Update the results into dictionary
    results['train_loss'] = train_loss
    results['train_acc'] = train_acc
    results['test_loss'] = test_loss
    results['test_acc'] = test_acc

  return results
