import torch
import torch.nn as nn
import torch.optim as optim
import os
from model import CNN
from utils import get_data_loaders, get_device

def train():
    # Detect GPU or fallback to CPU
    device = get_device()
    print(f"[*] Using device: {device}")

    # Initialize PyTorch DataLoaders
    train_loader, test_loader = get_data_loaders(batch_size=64)

    # Instantiate the CNN model and move it to the correct device (GPU/CPU)
    model = CNN().to(device)

    # Define the Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    print(f"[*] Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move tensors (images and labels) to the appropriate device
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass: compute predictions
            outputs = model(data)
            loss = criterion(outputs, targets)

            # Backward pass & Optimize
            optimizer.zero_grad() # Clear previous gradients
            loss.backward()       # Compute gradients backwards
            optimizer.step()      # Update weights

            running_loss += loss.item()

        # Print metrics after every epoch
        avg_train_loss = running_loss / len(train_loader)
        
        # Evaluate to check for overfitting alongside training
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # Save the trained model weights
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'cnn_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"[*] Model successfully saved to: {model_path}")

def evaluate(model, test_loader, device, criterion=None):
    model.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    test_loss = 0.0

    # Disable gradient calculation for faster inference & less memory usage
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            
            # Record test loss if a loss criterion was passed
            if criterion:
                loss = criterion(outputs, targets)
                test_loss += loss.item()

            # Find the predicted class (max probability)
            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader) if criterion else 0.0
    
    return avg_test_loss, accuracy

if __name__ == '__main__':
    train()
