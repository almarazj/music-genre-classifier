import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):   
    def __init__(self, input_shape):
        super(CNN, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)

        # Second convolution layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)

        # Third convolution layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 2))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(32)
        
        # Output size after last pooling layer
        self.flatten_size = self._get_flatten_size(input_shape)
        
        # Dense layer
        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.dropout = nn.Dropout(0.3)
        
        # Output layer
        self.fc2 = nn.Linear(64, 10)
    
    def _get_flatten_size(self, input_shape):
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            return x.numel()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.bn1(x)

        x = self.pool2(F.relu(self.conv2(x)))
        x = self.bn2(x)

        x = self.pool3(F.relu(self.conv3(x)))
        x = self.bn3(x)
        
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.fc2(x)
        return F.softmax(x, dim=1)