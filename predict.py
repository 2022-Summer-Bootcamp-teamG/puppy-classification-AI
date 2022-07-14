import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import json

use_cuda = torch.cuda.is_available()
use_pretrained = True
model_transfer = models.vgg16(pretrained=use_pretrained)

# Freeze weights
for param in model_transfer.features.parameters():
    param.required_grad = False

n_inputs = model_transfer.classifier[6].in_features

last_layer = nn.Linear(n_inputs, 20)
model_transfer.classifier[6] = last_layer

model_transfer.load_state_dict(torch.load('model_transfer.pt'))
model_transfer.eval()

if use_cuda:
    model_transfer = model_transfer.cuda()


def image_to_tensor(img_bytes):
    img = Image.open(img_bytes).convert('RGB')
    transformations = transforms.Compose([transforms.Resize(size=224),
                                          transforms.CenterCrop((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
    image_tensor = transformations(img)[:3, :, :].unsqueeze(0)
    return image_tensor


class_names = json.load(open('./imagenet_class_index.json'))


def predict_breed(img_bytes):
    # load the image and return the predicted breed
    image_tensor = image_to_tensor(img_bytes)

    # move model inputs to cuda, if GPU available
    if use_cuda:
        image_tensor = image_tensor.cuda()

    # get sample outputs
    output = model_transfer(image_tensor)
    _, preds_tensor = torch.max(output, 1)
    pred = np.squeeze(preds_tensor.numpy()) if not use_cuda else np.squeeze(preds_tensor.cpu().numpy())

    return class_names[str(pred)]
