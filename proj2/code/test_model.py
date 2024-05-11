import numpy as np
import torch
from net_class import Net

img_size = 50

net = Net()
net.load_state_dict(torch.load("melanoma_model.pth"))
#Evaluation mode
net.eval()

testing_data = np.load("testing_data.npy", allow_pickle=True)

