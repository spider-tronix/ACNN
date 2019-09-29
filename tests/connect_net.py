import torch

from utilities.connect_net import ConnectNet

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cnet = ConnectNet(kernel_size=(2, 2), device=device)
    input_img = torch.arange(18).reshape((2, 3, 3))  # input img
    filters = torch.arange(24).reshape(3, 2, 2, 2)  # input filters
    y_pred = cnet(input_img, filters)  # img convolved with filters
    print('Output is y_pred:\n', y_pred)
