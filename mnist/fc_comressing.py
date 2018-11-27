from tensortorch import easytt as ttm

import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Hyper-parameters
input_size = 784
hidden_size = 256
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

in_modes = [7,4,7,4]
out_modes = [2,8,8,2]
tt_ranks = [1,2,4,2,1]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=False)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


def num_model_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, compress=False, in_modes=None, out_modes=None, ranks=None):
        super(NeuralNet, self).__init__()
        if compress:
            self.fc1 = ttm.TTLayer(in_modes, out_modes, ranks)
        else:
            self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def train_nn(model, train_loader, criterion, optimizer, verbose=True):
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((i+1) % 100 == 0) and verbose:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    return model


def test_model(model, test_loader):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    return correct/total


model_tt = NeuralNet(input_size, hidden_size, num_classes,\
                     compress=True, in_modes=in_modes, out_modes=out_modes, ranks=tt_ranks).to(device)
criterion_tt = nn.CrossEntropyLoss()
optimizer_tt = torch.optim.Adam(model_tt.parameters(), lr=learning_rate)

model_tt = train_nn(model_tt, train_loader,criterion_tt, optimizer_tt)

model = NeuralNet(input_size, hidden_size, num_classes, compress=False).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model = train_nn(model, train_loader, criterion, optimizer)


print("Compressed model accuracy")
test_model(model_tt, test_loader)
print("Initial model accuracy")
test_model(model, test_loader)

print("Compress ratio {}".format(num_model_params(model)/num_model_params(model_tt)))