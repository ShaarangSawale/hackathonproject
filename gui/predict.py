import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import os
#from training import PATH

PATH = "C:/Users/antho/hackathonproject/gui/Face_Shape_Detection_Model.pt"
import torch
import torch.nn as nn
from torch import optim
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler
from sklearn.metrics import accuracy_score
from collections import OrderedDict

#loading the trained machine learning model
checkpoint = torch.load(PATH)
if checkpoint['model'] == "vgg16":
    model = models.vgg16(VGG16_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
else: print("Model not found")

model.class_to_idx = checkpoint['class_to_idx']
#print(checkpoint['class_to_idx'])

classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                                        ('relu', nn.ReLU()),
                                        ('Drop', nn.Dropout(0.5)),
                                        ('fc2', nn.Linear(5000, 6)),
                                        ('output', nn.LogSoftmax(dim=1))]))

model.classifier = classifier
model.load_state_dict = checkpoint['model_state_dict']

TOPK = 1
IMAGE_PATH = "C:/Users/antho/hackathonproject/gui/captured_image.jpg"

def process_image():
    image = Image.open(IMAGE_PATH)

    if image.size[0] > image.size[1]:
        image.thumbnail((5000, 256))
    else: 
        image.thumbnail((256, 5000))

    left_margin = (image.width-224)/2
    bottom_margin = (image.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    
    image = image.crop((left_margin, bottom_margin, right_margin, top_margin))

    np_image = np.array(image)/225
    mean = np.array([[0.485, 0.456, 0.406]]) 
    std = np.array([[0.229, 0.224, 0.225]])
    np_image = (np_image - mean) / std
    #np_image = np.transpose((2, 0, 1))
    return np_image

#determining a face-shape for the captured image
image1 = process_image()

image1 = torch.from_numpy(image1).type(torch.FloatTensor)
#print(image1)
#image_transformation = T.ToTensor()
#x = TF.to_tensor(image)

image1 = image1.unsqueeze_(0) # removing this changed error to complaining about size [3] instead of [1,3]
image1 = np.transpose(image1, (0, 3, 1, 2))

output = model.forward(image1)
probabilities = torch.exp(output)

top_probabilities, top_indices = probabilities.topk(TOPK)

top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0] 
top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 

idx_to_class = {value: key for key, value in model.class_to_idx.items()}

top_classes = [idx_to_class[index] for index in top_indices]

result = top_classes[0]

#print(result)

f = open("result.txt", "w")
f.write(result)

