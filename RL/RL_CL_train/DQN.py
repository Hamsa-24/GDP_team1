import numpy as np
import collections
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import random


class ReplayBuffer:
    ''' Experience Replay buffer '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)
    

class Qnet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, fc1, fc2, name='DQN_1'):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, fc1)
        self.fc2 = torch.nn.Linear(fc1, fc2)
        self.fc3 = torch.nn.Linear(fc2, action_dim)

        self.name = name

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def save(self):
        torch.save(self.state_dict(), os.path.join('models/CL', self.name))

    def load(self):
        self.load_state_dict(torch.load(os.path.join('models/CL', self.name)))



class DQN:
    ''' DQN algorithm '''
    def __init__(self, env, learning_rate, gamma,
                 epsilon, target_update, device, name, fc1=1024, fc2=512):
        self.action_dim = env.action_dim
        self.state_dim = env.state_dim
        self.q_net = Qnet(self.state_dim, self.action_dim, fc1, fc2, name=name).to(device)
        self.target_q_net = Qnet(self.state_dim, self.action_dim, fc1, fc2, name=name+'_target').to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().cpu().detach().numpy()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, self.action_dim).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

    def load_model(self):
        print('.... loading model ....')
        self.q_net.load()
        self.target_q_net.load()