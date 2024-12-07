
# AUTHOR: Gaurab Pokharel 
# EMAIL: gpokhare@gmu.edu 

import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Define LeNet-5 layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # 1 input channel, 6 output channels
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)            # 6 input channels, 16 output channels
        self.fc1_input_size = None  # This will be calculated dynamically
        self.fc1 = None  # Define later after calculating the input size
        self.fc2 = nn.Linear(120, 84)                                    # Fully connected layer
        self.fc3 = nn.Linear(84, 10)                                     # Fully connected layer (10 classes)
        
    @staticmethod
    def _initialize_weights(module):
        # Initialize weights for all layers
        if isinstance(module, nn.Conv2d):
            # Initialize convolutional layers with a normal distribution
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            # Initialize fully connected layers with a normal distribution
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Forward pass through convolutional and pooling layers
        x = F.tanh(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.tanh(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        # Dynamically calculate the input size for fc1
        if self.fc1 is None:
            self.fc1_input_size = x.view(x.size(0), -1).size(1)
            self.fc1 = nn.Linear(self.fc1_input_size, 120).to(x.device)

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
