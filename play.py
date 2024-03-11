from itertools import count
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from snake_env import Env
from dqn import DQN


# class DQN(nn.Module):
#     def __init__(self, n_observations, n_actions):
#         super(DQN, self).__init__()
#         self.layer1 = nn.Linear(n_observations, 128)
#         self.layer2 = nn.Linear(128, 128)
#         self.layer3 = nn.Linear(128, n_actions)
#
#     def forward(self, x):
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))
#         return self.layer3(x)


device = torch.device("cpu")

env = Env(render_mode="ansi", size=5)

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state.flatten())
model = DQN(n_observations, n_actions).to(device)
model.load_state_dict(torch.load(rf"models/{env.size}.pth"))


state, info = env.reset()
state = torch.tensor(state.flatten(), dtype=torch.float32, device=device).unsqueeze(0)
# total_reward = 0

for t in count():
    with torch.no_grad():
        action = model(state).max(1).indices.view(1, 1)

    observation, reward, terminated, truncated, info = env.step(action.item())
    if observation is not None:
        observation = observation.flatten()
    # reward = torch.tensor([reward], device=device)
    # total_reward += reward
    done = terminated or truncated

    if terminated or truncated:
        next_state = None
    else:
        next_state = torch.tensor(
            observation, dtype=torch.float32, device=device
        ).unsqueeze(0)

    state = next_state

    if done:
        break
    else:
        os.system("clear")
        print(env.render_state(info))
        time.sleep(0.1)
