import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Input shape: 1 channel (grayscale), 28x28 image size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)
        
        # After two max pooling layers, the 28x28 image becomes 7x7
        # 64 output channels * 7 * 7 spatial dimensions
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # 10 classes for digits 0-9

    def forward(self, x):
        # Pass through first conv & pooling layer
        x = self.pool(F.relu(self.conv1(x)))
        
        # Pass through second conv & pooling layer
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the tensor before fully connected layers
        x = torch.flatten(x, 1)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Output layer (no softmax here as we use CrossEntropyLoss)
        
        return x
