import os
import numpy as np
from PIL import ImageFile
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


def train(n_epochs, train_loader, valid_loader,
          model, optimizer, criterion, use_cuda, save_path):
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        # validate
        for batch_idx, (data, target) in enumerate(valid_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)

        print('Epoch: {}\tTraining Loss: {:.6f}\t Validation Loss: {:.6f}'.
              format(epoch, train_loss, valid_loss))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).    Saving model...'.
                  format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

    return model


def model_init(model):
    for param in model.features.parameters():
        param.required_grad = False

    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(n_inputs, 20)

    return model


ImageFile.LOAD_TRUNCATED_IMAGES = True

use_cuda = torch.cuda.is_available()
print(torch.cuda.is_available())

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([transforms.Resize(size=224),
                                transforms.CenterCrop((224, 224)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(10),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

data_dir = './data/dog_images/'
batch_size = 25

# Make dataloader
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform)
                  for x in ['train', 'valid', 'test']}

data_loaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], shuffle=True, batch_size=batch_size)
    for x in ['train', 'valid', 'test']}

class_names = image_datasets['train'].classes

### set param
model = models.vgg16(pretrained=True)
batch_size = 5
epochs = 20
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)
###

model = model_init(model)

if use_cuda:
    model = model.cuda()

model = train(epochs, data_loaders['train'], data_loaders['valid'], model,
              optimizer, criterion, use_cuda, 'best_model_wts.pt')  # 모델 학습 명령
