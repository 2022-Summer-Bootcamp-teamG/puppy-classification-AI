import os
import numpy as np
from PIL import ImageFile
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models


def test(loaders, model, criterion, use_gpu):
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)

        # update average test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
        # print testing statistics

    # calculate average loss
    test_loss = test_loss / len(loaders['test'].dataset)

    # print test statistics
    print('Testing Loss Average: {:.6f} '.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))


# select model
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

ImageFile.LOAD_TRUNCATED_IMAGES = True

use_cuda = torch.cuda.is_available()
print(torch.cuda.is_available())

batch_size = 25

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([transforms.Resize(size=224),
                                transforms.CenterCrop((224, 224)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(10),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Select data directory
data_dir = './data/'

# Make dataloader
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform)
                  for x in ['train', 'valid', 'test']}
loaders_scratch = {
    x: torch.utils.data.DataLoader(image_datasets[x], shuffle=True, batch_size=batch_size)
    for x in ['train', 'valid', 'test']}
data_loaders = loaders_scratch  # 왜?
class_names = image_datasets['train'].classes
criterion = nn.CrossEntropyLoss()

test(data_loaders, model_transfer, criterion, use_cuda)
