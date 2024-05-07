import torch.nn as nn
import torch.nn.functional as F

import torch 

from torch.utils.data import Dataset

import collections
import numpy as np

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'policy', 'q_value',
                               'value'])

class ExperienceBuffer:
    def __init__(self, capacity, device):
        self.buffer = collections.deque(maxlen=capacity)
        self.device = device

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        states, policy, q_value, value = \
            zip(*[self.buffer[idx] for idx in indices])
        
        states = torch.cat(states, dim=0).to(self.device).to(torch.float32)
        policy = torch.cat(policy, dim=0).to(self.device).to(torch.float32)
        q_value = torch.cat(q_value, dim=0).to(self.device).to(torch.float32)
        value = torch.cat(value, dim=0).to(self.device).to(torch.float32)

        return states, policy, q_value, value


class CustomDataset(Dataset):
    def __init__(self, states_path, policy_path, values_path, quant=True):
        # Cargamos los tensores desde archivos
        self.states = torch.load(states_path)
        self.policy = torch.load(policy_path)
        self.values = torch.load(values_path)
        self.quant = quant

    def __len__(self):
        # Asumimos que todos los archivos tienen la misma longitud
        return len(self.states)

    def __getitem__(self, idx):
        # Obtener un item por su índice
        state = self.states[idx].squeeze(0).to(torch.float32)
        policy = self.policy[idx].to(torch.float32)
        value = self.values[idx].to(torch.float32)
        # El target es una tupla que contiene policy y value
        if self.quant:
            return state, value
        else:
            return state, (policy, value)

class ResNet(nn.Module):
    def __init__(self, num_resBlocks, num_hidden, row_count, column_count, action_size, device=torch.device("cuda")):
        super().__init__()
        self.device = device 
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * row_count * column_count, action_size)
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * row_count * column_count, 1),
            nn.Tanh()
        )
        self.to(device)
        
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
        
        
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x