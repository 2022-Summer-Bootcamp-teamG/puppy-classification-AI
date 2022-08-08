import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import json
import routes.imageController as imageController
import os

use_cuda = False  # torch.cuda.is_available()
use_pretrained = True
model_transfer = models.resnet50(pretrained=True)
trained_weight = 'weights.pt'
classes_file = 'class_index.json'

# Freeze weights
# for param in model_transfer.features.parameters():
#     param.required_grad = False

# VGG
# n_inputs = model_transfer.classifier[6].in_features
#
# last_layer = nn.Linear(n_inputs, 20)
# model_transfer.classifier[6] = last_layer

num_ftrs = model_transfer.fc.in_features
model_transfer.fc = nn.Linear(num_ftrs, 132)

model_state_dict = torch.load(trained_weight, map_location=torch.device('cpu'))

class_names = json.load(open(classes_file))

model_transfer.load_state_dict(model_state_dict)
model_transfer.eval()

if use_cuda:
    model_transfer = model_transfer.cuda()


def image_to_tensor(img_name):
    img_bytes = imageController.get_image(img_name)
    img = Image.open(img_bytes).convert('RGB')
    transformations = transforms.Compose([transforms.Resize(size=224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])

    image_tensor = transformations(img).unsqueeze(dim=0)
    return image_tensor


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def probability(x):
    return softmax(x) * 100


def predict_breed(img_name):
    image_tensor = image_to_tensor(img_name)

    if use_cuda:
        image_tensor = image_tensor.cuda()

    output = model_transfer(image_tensor)
    output = output.cpu().detach().numpy()
    percentage = probability(output[0])
    index = np.where(np.array(percentage) >= 5.)[0]
    temp = []
    for i in index:
        temp.append(percentage[i])
    temp = np.argsort(temp)

    result = []
    for i in reversed(temp):
        i = index[i]
        result.append({'breed_id': str(i), 'percent': str(percentage[i]), 'breed': class_names[str(i)][1]})
    return result

