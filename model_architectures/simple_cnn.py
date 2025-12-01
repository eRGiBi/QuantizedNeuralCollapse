import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """A simple CNN for testing and trials with Batch Normalization, Dropout."""

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # --- Convolutional Layers with Batch Normalization (BN) ---

        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # Output size: (32, 16, 16) for a 32x32 input

        # Block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # Output size: (64, 8, 8) for a 32x32 input

        # Block 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # Output size: (128, 4, 4) for a 32x32 input

        self.flattened_size = 128 * 4 * 4  # = 2048

        self.flat = nn.Linear(self.flattened_size, 512)
        # self.dropout1 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)

        self.fc2 = nn.Linear(512, 512)

        self.fc1 = nn.Linear(512, 128)
        # self.dropout2 = nn.Dropout(0.1)

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))

        x = self.pool3(F.relu(self.bn5(self.conv5(x))))

        # Flatten
        # x = x.view(-1, self.flattened_size)
        x = torch.flatten(x, 1)

        x = F.relu(self.flat(x))
        # x = self.dropout1(x)

        p_features = F.relu(self.fc5(x))
        # p_features = F.relu(self.fc4(x))
        p_features = F.relu(self.fc3(x))
        p_features = F.relu(self.fc2(x))

        p_features = F.relu(self.fc1(x))
        # p_features = self.dropout2(p_features)

        output = self.classifier(p_features)

        return output
