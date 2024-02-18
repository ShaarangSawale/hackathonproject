import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import os


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

#defining variable device for better performace if a gpu is available on the local machine
device = "cuda" if torch.cuda.is_available() else "cpu"

#defining the transformations for the training, validation and testing datasets
train_transforms = T.Compose([
    T.Resize((224, 224)),
    T.RandomRotation(30),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

validation_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

BATCH_SIZE = 64
TRAINING_SET = "C:/Users/antho/hackathonproject/Masterset/Testset"
VALIDATION_SET = "C:/Users/antho/hackathonproject/Masterset/Trainset"
TESTING_SET = "C:/Users/antho/hackathonproject/Masterset/Validationset"

#defining the datasets
training_dataset = datasets.ImageFolder(TRAINING_SET, transform=train_transforms)
validation_dataset = datasets.ImageFolder(VALIDATION_SET, transform=validation_transforms)
testing_dataset = datasets.ImageFolder(TESTING_SET, transform=test_transforms)

#loading the datasets
train_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
validate_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(testing_dataset, batch_size=BATCH_SIZE)

import json

with open("Face_Shapes.json") as f:
    Face_Shapes = json.load(f)

model = models.vgg16(weights=VGG16_Weights.DEFAULT)

for parameter in model.parameters():
    parameter.requires_grad = False

from collections import OrderedDict

classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                                        ('relu', nn.ReLU()),
                                        ('Drop', nn.Dropout(0.5)),
                                        ('fc2', nn.Linear(5000, 6)),
                                        ('output', nn.LogSoftmax(dim=1))]))

model.classifier = classifier

LEARN_RATE = 0.001

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LEARN_RATE)

def validation(model, validate_loader, criterion):
    val_loss = 0
    accuracy = 0

    for images, labels in iter(validate_loader):
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        val_loss = val_loss + criterion(output, labels).item()
        probs = torch.exp(output)

        equality = (labels.data == probs.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

        return val_loss, accuracy

#prints = open("prints.txt", "a")

def classifier_trainer():
    epochs = 25
    steps = 0
    eval_per = 40

    model.to(device)

    for e in range(epochs):
        model.train()

        running_loss = 0

        for images, labels in iter(train_loader):
            steps = steps + 1
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss = running_loss + loss.item()

            if steps % eval_per == 0:

                model.eval()

                with torch.no_grad():
                    validation_loss, accuracy = validation(model, validate_loader, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/eval_per),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(validate_loader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validate_loader)))
            
                running_loss = 0
                model.train()
    
classifier_trainer()

def accuracy(model, test_loader):
    model.eval()
    model.to(device)

    with torch.no_grad():
        accuracy = 0

        for images, labels in iter(test_loader):
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            probs = torch.exp(output)

            equality = (labels.data == probs.max(dim=1)[1])
        
            accuracy += equality.type(torch.FloatTensor).mean()
        
        print("Test Accuracy: {}".format(accuracy/len(test_loader)))    
        
        
accuracy(model, test_loader)

PATH = "/Users/shaarangsawale/hackathonproject/Face_Shape_Detection_Model.pth"

def pickle(model):
    model.class_to_idx = training_dataset.class_to_idx

    checkpoint = {'model': "vgg16",
                  'class_to_idx': model.class_to_idx,
                  'model_state_dict': model.state_dict()
                 }
    torch.save(checkpoint, PATH)

pickle(model)







