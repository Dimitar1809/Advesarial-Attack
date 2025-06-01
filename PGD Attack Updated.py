import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

# PGD attack function implementing Madry et al. (2017) formulation
# The PGD attack generates adversarial examples by iteratively perturbing the input data
# with small steps while keeping the perturbation bounded by an epsilon ball around the original input.

def pgd_attack(model, criterion, spectrogram, label, epsilon=0.1, alpha=0.01, num_iter=40):
    # Clone the spectrogram to avoid modifying the original input
    # Adding random initialization for better exploration within the epsilon ball
    perturbed_spectrogram = spectrogram.clone().detach() + (torch.empty_like(spectrogram).uniform_(-epsilon, epsilon))
    perturbed_spectrogram = torch.clamp(perturbed_spectrogram, min=0, max=1).requires_grad_(True)

    for _ in range(num_iter):
        # Forward pass to compute the loss, measuring how far the model output is from the true label
        outputs = model(perturbed_spectrogram)
        loss = criterion(outputs, label)

        # Zero gradients to avoid accumulation from multiple iterations
        model.zero_grad()
        loss.backward()  # Compute gradients of loss w.r.t. the input

        # Apply the gradient step: move the input in the direction that increases the loss
        # Normalize gradients to prevent gradient explosion
        with torch.no_grad():
            gradient = perturbed_spectrogram.grad.sign()
            perturbed_spectrogram += alpha * gradient

            # Project back to the epsilon-ball around the original input to keep the perturbation controlled
            perturbation = torch.clamp(perturbed_spectrogram - spectrogram, min=-epsilon, max=epsilon)
            perturbed_spectrogram = torch.clamp(spectrogram + perturbation, min=0, max=1).detach()
            perturbed_spectrogram.requires_grad = True

    # Return the final adversarially perturbed example
    return perturbed_spectrogram

# Load pretrained model (example placeholder)
# This should be a trained CNN model for audio spectrogram classification
model = torch.load('alexnet_spectrogram_model.keras')
model.eval()

# Load AudioMNIST dataset (spectrogram based, example placeholder)
# The transform here converts data to tensors suitable for PyTorch
transform = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform), batch_size=1, shuffle=True)

# Define loss criterion
# CrossEntropyLoss measures the difference between predicted probabilities and true labels
criterion = nn.CrossEntropyLoss()

# Perform attack on a batch of the dataset
# We only attack a single batch for demonstration purposes
for batch in train_loader:
    spectrogram, label = batch
    # Generate the adversarial spectrogram using the PGD method
    perturbed_spectrogram = pgd_attack(model, criterion, spectrogram, label)
    break  # Stop after one batch for demonstration purposes

# Evaluate the attacked sample
# Observe if the model's prediction changes after the adversarial attack
output = model(perturbed_spectrogram)
predicted_label = torch.argmax(output, dim=1)
print(f"Original Label: {label.item()}, Predicted Label after attack: {predicted_label.item()}")
