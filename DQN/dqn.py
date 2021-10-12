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


class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer_limit = buffer_limit
        self.buffer = collections.deque(maxlen=self.buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)


class QValues:
    def __init__(self, n_obs, n_action, net_q, net_target, loss_fnc, optimizer, memory, is_gpu, eps_start=1, eps_end=0.1, r=0.99, gamma=0.9, lr=0.01):
        self.n_obs = n_obs  # observation space
        self.n_action = n_action  # action space

        self.net_q = net_q  # q network
        self.net_target = net_target # target network

        self.memory = memory # experience replay

        self.loss_fnc = loss_fnc
        self.optimizer = optimizer
        self.is_gpu = is_gpu 
        if self.is_gpu:
            self.net_q.cuda()
            self.net_target.cuda()

        self.eps = eps_start  # initial epsilon
        self.eps_end = eps_end  # lower bound of epsilon
        self.gamma = gamma  # discount rate
        self.r = r # decrement rate of epsilon
        self.lr = lr # learning rate

    def train(self, batch_size):
        for i in range(10):
            obs, action, reward, next_obs, done_mask = memory.sample(batch_size)

            if self.is_gpu:
                obs, action, reward, next_obs, done_mask = obs.cuda(), action.cuda(), reward.cuda(), next_obs.cuda(), done_mask.cuda()

            # current q-value in the online network
            q_out = self.net_q(obs) # batch_size x number of actions
            q_a = q_out.gather(1, action) # batch_size x 1 (select the corresponding q-value of each experience)

            # maximum q-value in the target network
            next_q_max = self.net_target(next_obs).max(1)[0].unsqueeze(1) # batch_size x 1

            # target
            target = reward + self.gamma * next_q_max * done_mask # batch_size x 1
            
            # difference between the current q-value and the target
            loss = self.loss_fnc(q_a, target) # 1 x 1
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
            q = self.net_q.forward(obs) # make decision with the q network
            action = np.argmax(q.detach().cpu().numpy())
            greedy = True
        return action, greedy
    
    def update_target(self):
        self.net_target.load_state_dict(self.net_q.state_dict())
    
    def update_eps(self):
        if self.eps > self.eps_end:
            self.eps *= self.r

