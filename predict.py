import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import os
from training import PATH


import torch
import torch.nn as nn
from torch import optim
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import DataLoader
import torchvision.transforms as T
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

classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                                        ('relu', nn.ReLU()),
                                        ('Drop', nn.Dropout(0.5)),
                                        ('fc2', nn.Linear(5000, 6)),
                                        ('output', nn.LogSoftmax(dim=1))]))

model.classifier = classifier
model.load_state_dict = checkpoint['model_state_dict']

TOPK = 1;
IMAGE_PATH = "/Users/shaarangsawale/hackathonproject/CapturedImage.jpg"

def process_image():
    image = Image.open(IMAGE_PATH)

    if image.size[0] > image.size[1]:
        image.thumbnail((5000, 256))
    else: 
        image.thumbnail((256, 5000))

    left_margin = (pil_image.width-224)/2
    bottom_margin = (pil_image.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    
    pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))

    np_image = np.array(image)/225
    mean = np.array(mean=[0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np.transpose((2, 0, 1))
    return np_image

#determining a face-shape for the captured image
image = process_image()

image = torch.from_numpy(image).type(torch.cuda.FloatTensor)

image = image.unsqueeze(0)

output = model.forward(image)
probabilities = torch.exp(output)

top_probabilities, top_indices = probabilities.topk(TOPK)

top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0] 
top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 

idx_to_class = {value: key for key, value in model.class_to_idx.items()}

top_classes = [idx_to_class[index] for index in top_indices]

result = top_classes[0]

print(result)










    
       

