import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
from torchviz import make_dot
import numpy as np

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     # Prefer GPU

np.random.seed(0)
torch.manual_seed(0)

# HyperParams and Others
num_epochs = 3
num_classes = 10
batch_size = 100
learning_rate = 0.01

# Loading Data
data_loc = '/media/syzygianinfern0/Summore Data/mnist_data'
t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST(root=data_loc, train=True, transform=t, download=True)
test_dataset = torchvision.datasets.MNIST(root=data_loc, train=False, transform=t)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class ParallelNets(nn.Module):
    """The neural network"""
    def __init__(self):
        """
        Init all variables for class
        """
        super(ParallelNets, self).__init__()
        self.net1conv1 = nn.Sequential(nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.net1conv2 = nn.Sequential(nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.net2conv1 = nn.Sequential(nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.net2conv2 = nn.Sequential(nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.net2conv3 = nn.Sequential(nn.Conv2d(20, 30, kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.net1fc1 = nn.Linear(28 * 28 * 20, 400)
        self.net2fc1 = nn.Linear(28 * 28 * 30, 600)
        self.dense = nn.Linear(1000, 10)

    def forward(self, x, y):
        """
        Forward prop implementation
        :param x: Input
        :param y: Input again
        :return: Predicted Labels
        """
        # Network 1
        xout = self.net1conv1(x)
        xout = self.net1conv2(xout)
        xout = xout.reshape(xout.size(0), -1)
        xout = self.net1fc1(xout)

        # Network 2
        yout = self.net2conv1(y)
        yout = self.net2conv2(yout)
        yout = self.net2conv3(yout)
        yout = yout.reshape(yout.size(0), -1)
        yout = self.net2fc1(yout)

        # Catenation
        out = torch.cat((xout, yout), dim=1)
        out = self.dense(out)
        return out


model = ParallelNets()
model.to(device)

# x = torch.zeros(1, 1, 28, 28, dtype=torch.float, requires_grad=False)
# out = model(x, x)
# make_dot(out).render("GradientFlowGraph", format="png")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

steps = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        images, labels = data[0].to(device), data[1].to(device)
        output = model(images, images)
        loss = criterion(output, labels)
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total = labels.size(0)
        _, predicted = torch.max(output.data, 1)        # Get prediction
        correct = (predicted == labels).sum().item()    # Evaluate
        acc_list.append(correct / total)
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {}%'.format(epoch + 1, num_epochs, i + 1, steps,
                                                                                    loss.item(),
                                                                                    (correct / total) * 100))

# Testing
model.eval()
with torch.no_grad():       # Dont track variables during evaluation
    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        output = model(images, images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

# Plotting with Bokeh
p = figure(y_axis_label='Loss', width=600, y_range=(0, 1), title='Results')
p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
p.line(np.arange(len(loss_list)), loss_list)
p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
show(p)
