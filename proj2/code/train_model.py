import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from net_class import Net

img_size = 50

#[img_array, [1,0]]
#Image array + classification of the image (benign or malignant)
training_data = torch.load("training_data.pt")

#Passing only the img data
train_X = torch.stack([item[0] for item in training_data])

#Normalize are values to make them between 0 and 1
#train_X = train_X / 255

#Passing only the classification that correspondes to each image in train_X
train_Y = torch.tensor([item[1] for item in training_data], dtype= torch.long)

net = Net()

#Adam optimizer
#lr = learning rate
optimizer = optim.Adam(net.parameters(), lr=0.001)

#Mean squared error loss function - commonly used in linear regression
loss_function = nn.CrossEntropyLoss()

#How many images that are passing through at once
batch_size = 100

#How many times we are passing through the training data
epochs = 10

for epoch in range(epochs):
    for i in range(0, len(training_data), batch_size):
        print(f"Epoch: {epoch+1}, {(i/len(train_X))*100}% complete")
        
        #The images have to be 1x50x50. the -1 is telling pytorch to be flexible in the input images.
        batch_X = train_X[i: i+batch_size]
        batch_Y = train_Y[i: i+batch_size]
        
        #Resetting the gradients of model parameters to zero
        optimizer.zero_grad()

        #e.g.[0.34, 0.66]
        outputs = net(batch_X)

        #Value of the loss for this batch. Between output and actual image.
        loss = loss_function(outputs, batch_Y)

        #Back propagation done to calculate gradients of the loss with respect to model parameters and move in the "opposite direction"
        loss.backward()

        #Optimizer updates the model params based on the gradient we just calculated
        optimizer.step()

    print(f"Epoch: {epoch+1}, 100.0% complete")

torch.save(net.state_dict(), "melanoma_model.pth")
