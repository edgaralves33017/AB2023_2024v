import torch
from torchvision import transforms
import os
import cv2
import numpy as np

#this is because images can be diferent sizes, so we scale them to be the same size
#50x50 pixels
image_size = 50

#locations of image files
benign_training_folder = "../dataset/train/benign/"
malignant_training_folder = "../dataset/train/malignant/"

benign_testing_folder = "../dataset/test/benign/"
malignant_testing_folder = "../dataset/test/malignant/"

#Saving the training + testing data in arrays
benign_training_data = []
malignant_training_data = []

benign_testing_data = []
malignant_testing_data = []

# Transformações de aumento de dados
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def addImagesToArray(listToAdd, folder, code):
    for filename in os.listdir(folder):
        try:
            path = folder+filename
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            #Resize image
            img = cv2.resize(img, (image_size,image_size))
            img_tensor = data_transforms(img)
            #2nd parameter: Code for if its benign ( 0 ) or malignant ( 1 )
            listToAdd.append([img_tensor, code])
        except:
            pass


#Adding the images to their respective arrays
addImagesToArray(benign_training_data, benign_training_folder, 0)
addImagesToArray(benign_testing_data, benign_testing_folder, 0)
addImagesToArray(malignant_training_data, malignant_training_folder, 1)
addImagesToArray(malignant_testing_data, malignant_testing_folder, 1)

#Make sure there isn't an imbalance of training data to prevent the model from being biased.
#Cut vectors from the shortest length.
shortestLength = len(benign_training_data) if len(benign_training_data) < len(malignant_training_data) else len(malignant_training_data)
benign_training_data = benign_training_data[0:shortestLength]
malignant_training_data = malignant_training_data[0:shortestLength]

shortesttestingLength = len(benign_testing_data) if len(benign_testing_data) < len(malignant_testing_data) else len(malignant_testing_data)

benign_testing_data = benign_testing_data[0:shortesttestingLength]
malignant_testing_data = malignant_testing_data[0:shortesttestingLength]



#Check if there is an imbalance on data so that the model is not biased
print()
print(f"Benign training count: {len(benign_training_data)}")
print(f"Malignant training count: {len(malignant_training_data)}")
print()
print(f"Benign testing count: {len(benign_testing_data)}")
print(f"Malignant testing count: {len(malignant_testing_data)}")

#Merging training data
training_data = benign_training_data+malignant_training_data
#Shuffling the data so it isn't 100 benign images followed by 100 malignant images
np.random.shuffle(training_data)
torch.save(training_data,"training_data.pt")

#Merging testing data.
testing_data = benign_testing_data+malignant_testing_data
#Shuffling the data so it isn't 100 benign images followed by 100 malignant images
np.random.shuffle(testing_data)
torch.save(testing_data, "testing_data.pt")
