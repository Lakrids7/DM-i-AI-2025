import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()

        # Define the network architecture
        # Input: 16 sensor readings
        # Output: 5 values, one for each action, representing the probability
        # of taking that action.
        self.fc1 = nn.Linear(16, 128) # 16 inputs, 128 hidden units
        self.dropout = nn.Dropout(p=0.5) # Dropout to prevent overfitting
        self.fc2 = nn.Linear(128, 5)  # 128 hidden units, 5 outputs

    def forward(self, state):
        """
        Performs the forward pass through the network.
        Args:
            state (torch.Tensor): The current state of the environment (sensor data).
        Returns:
            torch.Tensor: The probabilities for each action.
        """
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = self.fc2(x)
        # We use softmax to convert the network's raw output scores into a
        # probability distribution over the possible actions.
        # This ensures the outputs sum to 1 and can be interpreted as probabilities.
        return F.softmax(x, dim=-1)