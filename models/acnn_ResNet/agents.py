import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class AcnnResNet(nn.Module):

    def __init__(self, cn_kernel_size=(3, 3), cn_stride=1,
                 device='cuda:0'):
        """
        Init all variables for class
        """

        super(AcnnResNet, self).__init__()
        self.device = device

        # ----------------------Features Network------------------------#
        
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1)

        self.block1 = nn.Sequential(
             nn.Conv2d(64, 64, 3, stride=1, padding=1),
             nn.ReLU(),
             nn.Conv2d(64, 64, 3, stride=1, padding=1)             
        )

        self.block2 = nn.Sequential(
             nn.Conv2d(64, 64, 3, stride=1, padding=1),
             nn.ReLU(),
             nn.Conv2d(64, 64, 3, stride=1, padding=1)         
        )

        self.conv6 = nn.Conv2d(64, 128, 3, stride=1)

        self.block3 = nn.Sequential(
             nn.Conv2d(128, 128, 3, stride=1, padding=1),
             nn.ReLU(),
             nn.Conv2d(128, 128, 3, stride=1, padding=1)             
        )

        self.conv9 = nn.Conv2d(128, 256, 3, stride=1)

        self.block4 = nn.Sequential(
             nn.Conv2d(256, 256, 3, stride=1, padding=1),
             nn.ReLU(),
             nn.Conv2d(256, 256, 3, stride=1, padding=1)             
        )

        #---------------------Filters Network-------------------------#   

        self.net2 = nn.Sequential(
            nn.Conv2d(1, 16 ,3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.ReLU()
        )

        #--------------------Connect Network--------------------------#

        self.connect_net = ConnectNet(cn_kernel_size,
                                      strides=cn_stride,
                                      device=self.device)
        
        #--------------------Classifier Network-----------------------#

        self.classifier = nn.Sequential(
            #nn.Linear(19*19*64, 1024),
            nn.Linear(22*22*256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.LogSoftmax()
        )

    def forward(self, X_in):
        """
        forward propagation logic
        """

        #---------------------features network------------------------------#
        x = self.conv1(X_in)
        residual1 = x   # save input as residual
        x = self.block1(x)

        x += residual1  # add residual to output of block 1
        x = F.relu(x)  # perform relu non-linearity
        residual2 = x   # update residual
        x = self.block2(x)

        x += residual2 
        x = F.relu(x) 
        x = self.conv6(x)

        residual3 = x   # update residual
        x = self.block3(x)
        
        x += residual3
        x = F.relu(x)
        x = self.conv9(x)

        residual4 = x
        x = self.block4(x)
        x += residual4
        out1 = F.relu(x)
    
        #---------------------filters network------------------------------#

        out2 = self.net2(X_in)  # not x, but X_in (raw input)

        #---------------------Connect network-------------------------------#
        
        batch_size, c_in, h, w = out1.shape
        _, c_out, kh, kw = out2.shape
        new_h, new_w = h - kh + 1, h - kw + 1

        out3 = torch.zeros((batch_size, c_out, new_h, new_w),
                           device=self.device)  

        for i in range(batch_size):
            i_out2 = torch.squeeze(out2[i])[:, None, :, :]
            i_out2 = i_out2.repeat(1, c_in, 1, 1)  # broadcasting
            out3[i] = self.connect_net.forward(out1[i], i_out2)
        
        # TODO: Perform Relu operation on connect net's output        
        #---------------------Classifier network----------------------------#
        
        #out3 = out3.reshape(batch_size, -1)
        out3 = out1.reshape(batch_size, -1)
        out3 = self.classifier(out3)
        return out3