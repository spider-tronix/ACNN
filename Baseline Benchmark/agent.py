from torch import nn


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 10, 3),
            nn.ReLU(),
            nn.Conv2d(10, 20, 3),
            nn.ReLU(),
            nn.Conv2d(20, 30, 5),
            nn.ReLU(),
            nn.Conv2d(30, 40, 5),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(40*16*16, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            # nn.Softmax()
        )

    def forward(self, X):
        out = self.convs(X)
        out = self.fc(out.view(-1, 40*16*16))
        return out
