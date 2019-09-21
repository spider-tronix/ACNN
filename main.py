from agents import Viba
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Prefer GPU

np.random.seed(0)
torch.manual_seed(0)

# HyperParams and Others
num_epochs = 3
num_classes = 10
batch_size = 1
learning_rate = 0.01

model = Viba()

if __name__ == '__main__':
    # Loading Data
    data_loc = '/media/syzygianinfern0/Summore Data/Datasets'
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
    train_dataset = torchvision.datasets.MNIST(root=data_loc, train=True, transform=t, download=True)
    test_dataset = torchvision.datasets.MNIST(root=data_loc, train=False, transform=t)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = Viba()
    model.to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    steps = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            images, labels = data[0].to(device), data[1].to(device)
            output = model(images)
            loss = criterion(output, labels)
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total = labels.size(0)
            _, predicted = torch.max(output.data, 1)  # Get prediction
            correct = (predicted == labels).sum().item()  # Evaluate
            acc_list.append(correct / total)
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {}%'.format(epoch + 1, num_epochs, i + 1,
                                                                                        steps,
                                                                                        loss.item(),
                                                                                        (correct / total) * 100))
    # Testing
    model.eval()
    with torch.no_grad():  # Dont track variables during evaluation
        correct = 0
        total = 0
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            output = model(images, images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))