import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import os


import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.cuda.amp import GradScaler
from sklearn.metrics import accuracy_score

from torchvision.models import resnet101, ResNet101_Weights
weights = ResNet101_Weights
preprocess = weights.transforms()
preprocess
resnet = resnet101(weights=weights)
resnet
from PIL import Image
image= Image.open('\face shape detector\diamond\download (1).jpg') 

