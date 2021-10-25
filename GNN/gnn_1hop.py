# !pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
# !pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
# !pip install -q torch-geometric

import numpy as np
import collections
import random

import networkx as nx
from scipy.spatial import distance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch_geometric.nn import GCNConv
from torch_geometric.nn import Linear
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def convert_to_graph(observations, positions, boundary, n_agents):
    # convert observations
    x = torch.tensor(observations, dtype=torch.float)

    # compute distances between drones
    mat_distance = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(n_agents):
            mat_distance[i][j] = distance.euclidean(positions[i], positions[j])
    
    # convert to adjacency matrix
    mask = mat_distance <= boundary
    mat_adjacency = mask * 1

    # replace diagonal elements
    mat_adjacency[range(n_agents), range(n_agents)] = 0

    # convert adjacency matrix to coo format 
    edges = np.where(mat_adjacency == 1)
    edge_index = torch.tensor([edges[0], edges[1]], dtype=torch.long)

    # convert to PyTorch Geometric Data
    data = Data(x=x, edge_index=edge_index)

    return data


class QNet(nn.Module):
    def __init__(self, n_obs, n_mid, n_action):
        super(QNet, self).__init__()
        self.conv1 = GCNConv(n_obs, n_mid)
        self.fc1 = Linear(n_mid, n_action)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.tanh()
        x = self.fc1(x)
        return x

    def sample_action(self, data, epsilon):
        out = self.forward(data) # torch.Size([n_agents, n_action])
        out = torch.reshape(out, (1, out.shape[0], out.shape[1])) # torch.Size([1, n_agents, n_action])
        mask = (torch.rand((out.shape[0],)) <= epsilon)  # torch.Size([1])
        action = torch.empty((out.shape[0], out.shape[1],)) # torch.Size([n_agents, n_action])
        action[mask] = torch.randint(0, out.shape[1], action[mask].shape).float()
        action[~mask] = out[~mask].argmax(dim=2).float()
        return action