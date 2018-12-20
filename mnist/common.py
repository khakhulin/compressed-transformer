import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


def get_device():
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mnist_loaders(root, batch_size):
  def _get_dataset(train):
    return torchvision.datasets.MNIST(
      root=root, train=train,
      transform=transforms.ToTensor(),
      download=True
    )
  train, test = _get_dataset(True), _get_dataset(False)

  def _get_loader(data, shuffle):
    return torch.utils.data.DataLoader(
      data, batch_size=batch_size, shuffle=shuffle)

  train, test = _get_loader(train, True), _get_loader(test, False)
  return train, test


class MNISTModel(nn.Module):
  def __init__(self, units=None):
    super(MNISTModel, self).__init__()
    if units is None:
      units = [784, 256, 10]
    layers = []
    for in_features, out_features in zip(units[:-1], units[1:]):
      layers.append(nn.Linear(in_features, out_features))
      layers.append(nn.ReLU())
    layers.pop()
    self.layers = nn.ModuleList(layers)
    self.to(get_device())

  def forward(self, inputs):
    x = inputs.reshape(-1, 28 * 28).to(get_device())
    for layer in self.layers:
      x = layer(x)
    return x


def learn(model, trainloader, nepochs, optimizer,
          metrics=None, logperiod=10):
  metrics = metrics or []
  device = get_device()

  datasize = len(trainloader)
  for epoch in range(nepochs):
    iters = []
    losses = []
    metrics_values = [[] for _ in enumerate(metrics)]
    for i, (inputs, labels) in enumerate(trainloader):
      outputs = model(inputs)
      labels = labels.to(device)

      loss = nn.functional.cross_entropy(outputs, labels)
      if i % logperiod == 0:
        iters.append(datasize * epoch + i)
        losses.append(loss)
        for i, metric in enumerate(metrics):
          metrics_values[i].append(metric(outputs, labels))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    yield iters, losses, metrics_values


def accuracy(outputs, labels):
  _, predictions = torch.max(outputs, -1)
  return (labels == predictions).type(torch.float32).mean()


def nparams(model, nnz=False):
  params = (p for p in model.parameters() if p.requires_grad)
  return sum([np.prod(p.size() if not nnz else (p != 0).sum().item())
              for p in params])


def update_line(line, newxs, newys):
  xs, ys = line.get_data()
  xs.extend(newxs)
  ys.extend(newys)
  line.set_data(xs, ys)
  line.axes.relim()
  line.axes.autoscale_view(True)
