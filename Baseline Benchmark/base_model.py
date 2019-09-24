import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 64, 3)
        self.fc1 = nn.Linear(64*24*24, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x))


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # doesn't track operations for performing backprop
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



if name == '__main__':

	batch_size = 64
	lr = 0.01
	epochs = 3
	
	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	data_loc = '/content/'

	t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
	train_dataset = datasets.MNIST(root=data_loc, train=True, transform=t, download=True)
	test_dataset = datasets.MNIST(root=data_loc, train=False, transform=t)
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

	model = BaseModel().to(device)
	optimizer = optim.SGD(model.parameters(), lr=lr)

	for epoch in range(1, epochs + 1):
	    train(model, device, train_loader, optimizer, epoch, log_interval=100)
	    test(model, device, test_loader)
