# -*- coding: utf-8 -*-
"""
ARS2: Robotic Vision
Fall 2020

Practice session 02: CNN

@author: philippe.xu@hds.utc.fr
"""

import torch
# pretrained dataset
from torchvision import transforms, models
# index to lebal of imagenet dataset
from imagenet_labels import idx_to_label
# image processing package
from PIL import Image
import os

# Load the VGG16 pretrained CNN
vgg16 = models.vgg16(pretrained=True)

# Read an image
# input_image = Image.open(os.path.join('cat', 'cat0.jpg'))
input_image = Image.open(os.path.join('dataset', 'Federico_Faruffini', 'down', '11.jpg'))
# Preprocessing chain: resize, crop, tensor, normalize
preprocess = transforms.Compose([
    transforms.Resize(256),  # a little bit bigger to not loss border
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # necessary for neural network
])

# Preprocess the image
input_tensor = preprocess(input_image)

# Build a tensor batch
input_batch = input_tensor.unsqueeze(0)

# Predict the output without computing the gradient
with torch.no_grad():
    output = vgg16(input_batch)

# Normalize the output as a probability distribution with a softmax
prob = torch.nn.functional.softmax(output[0], dim=0)  # not the prob in the sense of statistic

# Get the k highest scores from the prediction
topk = torch.topk(prob, 5)
topk_prob = topk.values.numpy()
topk_idx = topk.indices.numpy()

# Print the labels of the topk classes
for i in range(len(topk_prob)):
    print(f'Class: {idx_to_label[topk_idx[i]]}, probability={topk_prob[i]:2.2f}')
