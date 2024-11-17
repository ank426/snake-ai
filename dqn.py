import torch
import torch.nn as nn


# class DQN(nn.Module):
#     def __init__(self, n_observations, n_actions):
#         super(DQN, self).__init__()
#         self.n_observations = n_observations
#         self.n_actions = n_actions
#
#     def forward(self, x):
#         x = torch.reshape(x, (x.shape**0.5, x.shape**0.5))
#         x = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=2, stride=1)(x)
#         x = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)(x)
#         x = nn.ReLU()(x)
#         x = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=2, stride=1)(x)
#         x = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)(x)
#         x = nn.ReLU()(x)
#         x = nn.Flatten()(x)
#         x = nn.Linear(x.shape[0], self.n_actions)(x)
#         return x


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.n_observations = n_observations
        self.layers = nn.Sequential(
            # 1, 16, 16
            nn.Conv2d(
                in_channels=1, out_channels=4, kernel_size=2, stride=1, padding=1
            ),
            # 4, 16, 16
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 4, 8, 8
            nn.ReLU(),
            nn.Conv2d(
                in_channels=4, out_channels=4, kernel_size=2, stride=1, padding=1
            ),
            # 4, 8, 8
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 4, 4, 4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        x = x.reshape(
            (-1, 1, int(self.n_observations**0.5), int(self.n_observations**0.5))
        )
        return self.layers(x)


# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class DQN(nn.Module):
#     def __init__(self, n_observations, n_actions):
#         super(DQN, self).__init__()
#         self.layer1 = nn.Linear(n_observations, 128)
#         self.layer2 = nn.Linear(128, 128)
#         self.layer3 = nn.Linear(128, n_actions)
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = F.relu(x)
#         x = self.layer2(x)
#         x = F.relu(x)
#         x = self.layer3(x)
#         return x
