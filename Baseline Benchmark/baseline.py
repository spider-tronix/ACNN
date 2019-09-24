import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import torch.optim as optim
from agent import Baseline
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Baseline().to(device)

# HyperParams and Others
EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.01


def load_data():
    """
    Loads the MNIST data
    :return: Train and test Data of type torch.utils.data.dataloader
    """
    data_loc = 'E:\Datasets'
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root=data_loc, train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root=data_loc, train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader


def evaluate(dataloader: torch.utils.data.dataloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct / total) * 100


def train(trainloader: torch.utils.data.dataloader,
          testloader: torch.utils.data.dataloader):
    """
    Training Loop
    :param testloader:
    :param trainloader:
    :return:
    """

    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        no_steps = len(trainloader)
        running_loss = 0.0
        loss_list = []
        acc_list = []
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Loggers
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / BATCH_SIZE)
            loss_list.append(loss.item())
            running_loss += loss.item()

            # Print Statistics
            if (i + 1) % 200 == 0:  # print every 200 mini-batches
                print(f"Epoch: [{epoch + 1}/{EPOCHS}]  Batch: [{i + 1}/{no_steps}]   "
                      f"Batch Train Loss: {running_loss / BATCH_SIZE}")
                running_loss = 0.0
        val_acc = evaluate(testloader)
        train_acc = evaluate(trainloader)
        print(f"Train Accuracy: {train_acc}    Validation Accuracy: {val_acc} \n")


if __name__ == '__main__':
    train_data, test_data = load_data()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train(train_data, test_data)
