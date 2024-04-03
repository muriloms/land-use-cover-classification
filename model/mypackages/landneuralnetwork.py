

import torch
import torch.nn as nn
import torch.nn.functional as F

# Defines a new model class named LandClassifierNet, inheriting from nn.Module
class LandClassifierNet(nn.Module):
    # Constructor method of the class
    def __init__(self):
        # Calls the constructor of the parent class (nn.Module)
        super(LandClassifierNet, self).__init__()

        # Defines the first convolutional layer with 3 input channels, 64 output channels,
        # and a kernel size of 3x3. This layer is responsible for capturing low-level features
        # such as edges and textures from the input images.
        self.conv1 = nn.Conv2d(3, 64, 3, 1)

        # Defines the second convolutional layer with 64 input channels, 128 output channels,
        # and a kernel size of 3x3. This layer further processes features extracted by the
        # previous layer, capturing more complex patterns.
        self.conv2 = nn.Conv2d(64, 128, 3, 1)

        # Defines the third convolutional layer with 128 input channels, 256 output channels,
        # and a kernel size of 3x3. Each subsequent convolutional layer increases the ability
        # of the network to represent higher-level features.
        self.conv3 = nn.Conv2d(128, 256, 3, 1)

        # Defines the first dropout layer with a dropout probability of 0.25.
        # Dropout is a regularization technique to prevent overfitting by randomly
        # setting a fraction of input units to 0 during training.
        self.dropout1 = nn.Dropout(0.25)

        # Defines the second dropout layer with a dropout probability of 0.5.
        # Higher dropout rates are typically used in layers closer to the output.
        self.dropout2 = nn.Dropout(0.5)

        # Defines the first fully connected (Dense) layer that maps from 215296 to 2048 neurons.
        # The fully connected layers act as classifiers on top of the features extracted by the convolutional layers.
        self.fc1 = nn.Linear(215296, 2048)

        # Defines the second fully connected layer that maps from 2048 to 512 neurons.
        self.fc2 = nn.Linear(2048, 512)

        # Defines the third fully connected layer that maps from 512 to 128 neurons.
        self.fc3 = nn.Linear(512, 128)

        # Defines the fourth fully connected layer that maps from 128 to 10 neurons.
        # The output size of 10 corresponds to the number of classes in the classification task.
        self.fc4 = nn.Linear(128, 10)

    # Defines the forward method for the forward pass
    def forward(self, x):
        # Applies the convolutional layer abd tge ReLU activation function
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        # Applies max pooling with a kernel size of 2x2
        # Max pooling reduces the spatial dimensions (height, width) of the input volume.
        x = F.max_pool2d(x, 2)

        # Applies the first dropout layer
        x = self.dropout1(x)

        # Flattens the tensor to prepare it for the fully connected layer
        x = torch.flatten(x, 1)

        # Applies the first fully connected layer
        x = self.fc1(x)
        x = F.relu(x)

        # Applies the second dropout layer
        x = self.dropout2(x)

        # Applies the second fully connected layer
        x = self.fc2(x)
        x = F.relu(x)

        # Applies the third fully connected layer
        x = self.fc3(x)
        x = F.relu(x)

        # Applies the fourth fully connected layer
        x = self.fc4(x)

        # Returns the log-softmax of the resulting tensor along dimension 1
        # Log-softmax is typically used for classification tasks as it provides
        # a probability distribution over classes.
        return F.log_softmax(x, dim=1)


