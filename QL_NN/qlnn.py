import numpy as np
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class Net(nn.Module):
    def __init__(self, n_obs, n_mid, n_action):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, n_mid) 
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class QValues:
    def __init__(self, n_obs, n_action, net, loss_fnc, optimizer, is_gpu, eps=1, gamma=0.9, r=0.99, lr=0.01):
        self.n_obs = n_obs  # observation space
        self.n_action = n_action  # action space

        self.net = net  # neural network model
        self.loss_fnc = loss_fnc
        self.optimizer = optimizer
        self.is_gpu = is_gpu 
        if self.is_gpu:
            self.net.cuda()

        self.eps = eps  # initial epsilon
        self.gamma = gamma  # discount rate
        self.r = r # decrement rate of epsilon
        self.lr = lr # learning rate

    def train(self, obs, next_obs, action, reward, done):

        obs = torch.from_numpy(obs).float()
        next_obs = torch.from_numpy(next_obs).float()
        if self.is_gpu:
            obs, next_obs = obs.cuda(), next_obs.cuda()
            
        self.net.eval()
        next_q = self.net.forward(next_obs)
        self.net.train()
        q = self.net.forward(obs)

        t = q.clone().detach()
        if done:
            t[action] = reward 
        else:
            t[action] = reward + self.gamma * np.max(next_q.detach().cpu().numpy())
            
        loss = self.loss_fnc(q, t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, obs):
        obs = torch.from_numpy(obs).float()
        if self.is_gpu:
            obs = obs.cuda()

        if np.random.rand() < self.eps:
            action = np.random.randint(self.n_action)
            greedy = False
        else: 
            q = self.net.forward(obs)
            action = np.argmax(q.detach().cpu().numpy())
            greedy = True
        return action, greedy
    
    def update_eps(self):
        if self.eps > 0.1:
            self.eps *= self.r