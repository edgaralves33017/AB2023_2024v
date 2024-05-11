import torch.nn as nn #Neural Network
import torch.nn.functional as f

#50x50 pixels
img_size = 50

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        #Convolutional Layers
        #last value has to be the first in the following layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)

        #Fully Connected Layer
        #First value in this "fc1" is -- check last print
        self.fc1 = nn.Linear(128*2*2, 512)
        #
        #Final Fully Connected Layer has 2 in the final parameter
        #due to this neural network being a binary classifier (benign ou malignant)
        self.fc2 = nn.Linear(512, 2)

    #Method has to be called "forward"
    #How the image "moves through" the network
    def forward(self, x):
        #RELU = Linear Regression: If num is positive, the number stays; if num is negative, makes it 0
        #Max Pooling 2d = transforms the output into a "smaller output". The (2,2) is the kernel size.
        x = f.max_pool2d(f.relu(self.conv1(x)), (2,2))
        #print(f"shape after conv1: {x.shape}")
        x = f.max_pool2d(f.relu(self.conv2(x)), (2,2))
        #print(f"shape after conv2: {x.shape}")
        x = f.max_pool2d(f.relu(self.conv3(x)), (2,2))
        #!! This print is important: It shows what the first value for "self.fc1" should be
        #e.g. "shape after conv3: torch.Size[1,128,2,2]" - The value for the self.fc1 will be 128*2*2
        #print(f"shape after conv3: {x.shape}")

        #"Flatten" the values after conv3
        x = x.view(-1, 128*2*2)
        
        #Pass through fc1
        x = f.relu(self.fc1(x))
        #Pass through fc2
        x = self.fc2(x)

        #Transforms value into a probability distribution
        x = f.softmax(x, dim=1)

        return (x)
