import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CifarResNet(nn.Module):
    """
        All in one ResNet model for training on CIFAR dataset.
        Implements ResNet-20, ResNet-32, ResNet-44, ResNet-56, ResNet-110.
    """

    def __init__(self, n):
        """
        Init all variables for class
        """
        super(CifarResNet, self).__init__()

        if n not in [3, 5, 7, 9, 11]:
            raise NotImplementedError('Resnet 20/32/44/56/110 only implemented. \
                                            Contact Sharan or Sachin, if u need any other model')
        self.n = n

        l = 1  # keeps count of layers added
        self.relu_list = np.zeros(6*n+2)

        self.model = nn.Sequential()
        self.model.add_module(f'conv{l}', nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False))
        self.model.add_module(f'bn{l}', nn.BatchNorm2d(16))
        self.model.add_module(f'relu{l}', nn.ReLU())
        self.relu_list[l] = 1
        l += 1
        

        for _ in range(n):
            self.model.add_module(f'conv{l}', nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=False))
            self.model.add_module(f'bn{l}', nn.BatchNorm2d(16))
            self.model.add_module(f'relu{l}', nn.ReLU())
            self.relu_list[l] = 1
            l += 1
            self.model.add_module(f'conv{l}', nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=False))
            self.model.add_module(f'bn{l}', nn.BatchNorm2d(16))
            l += 1
            
        
        for i in range(n):
            if i == 0:
                self.model.add_module(f'conv{l}', nn.Conv2d(16, 32, 1, stride=2, padding=1, bias=False)) # k_size = 1, stride = 2 for downsampling
            else:
                self.model.add_module(f'conv{l}', nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False)) # k_size = 3, stride = 1
            self.model.add_module(f'bn{l}', nn.BatchNorm2d(32))
            self.model.add_module(f'relu{l}', nn.ReLU())
            self.relu_list[l] = 1
            l += 1
            self.model.add_module(f'conv{l}', nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False))
            self.model.add_module(f'bn{l}', nn.BatchNorm2d(32))
            l += 1
                  

        for i in range(n):
            if i == 0:
                self.model.add_module(f'conv{l}', nn.Conv2d(32, 64, 1, stride=2, padding=1, bias=False)) # k_size = 1, stride = 2 for downsampling
            else:
                self.model.add_module(f'conv{l}', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)) # k_size = 3, stride = 1
            self.model.add_module(f'bn{l}', nn.BatchNorm2d(64))
            self.model.add_module(f'relu{l}', nn.ReLU())
            self.relu_list[l] = 1
            l += 1
            self.model.add_module(f'conv{l}', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False))
            self.model.add_module(f'bn{l}', nn.BatchNorm2d(64))
            l += 1
            
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Sequential(
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )


    # noinspection PyPep8Naming
    def forward(self, X):
        """
        forward propagation logic
        """

        layers = {name:module for name, module in self.model.named_modules()}
        id_dict = {f'identity{i}': None for i in range(3*self.n)}    # network will have 3n skip connections

        out = layers['conv1'](X)
        out = layers['bn1'](out)
        out = layers['relu1'](out)

        k = 0 
        for i in range(2, 6 * self.n + 2):
            id_dict[f'identity{k}'] = out   # residual
            out = layers[f'conv{i}'](out)
            out = layers[f'bn{i}'](out)
            if self.relu_list[i] == 1: 
                out = layers[f'relu{i}'](out)
            if self.relu_list[i] == 0:
                out += id_dict[f'identity{k}'] # residual added
                out = F.relu(out)
                k += 1
            
            # Uncomment to verify    
            #if i == 2*self.n + 2 or i == 4*self.n + 2: print('-'*10) # Basically print "-" after 2n+1, 4n + 1 (2n + 2n + 1)
            #print(out.shape, '\t channels', out.shape[1], '\t layer', i) 

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    n = 3 # try 5, 7, 9, 11
    model = CifarResNet(n)
    print(model(torch.rand((1, 3, 32, 32))))
