import torch
import torch.nn.functional as F
import numpy as np
import os


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, high_action, device, fc1, fc2, name='model1'):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, fc1)
        self.fc2 = torch.nn.Linear(fc1, fc2)
        self.fc3 = torch.nn.Linear(fc2, 2*action_dim)

        self.name = name
        self.action_dim = action_dim
        self.high_action = high_action
        self.device = device

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        mus = torch.tanh(x[:,:self.action_dim])*torch.tensor([self.high_action]).to(self.device)
        sigmas = torch.nn.functional.softplus(x[:,self.action_dim:]) + 1e-5
        return torch.cat((mus, sigmas), dim=1)
    
    def save(self):
        torch.save(self.state_dict(), os.path.join('models/nav', self.name))

    def load(self):
        self.load_state_dict(torch.load(os.path.join('models/nav', self.name)))
    

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, fc1, fc2):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, fc1)
        self.fc2 = torch.nn.Linear(fc1, fc2)
        self.fc3 = torch.nn.Linear(fc2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class ActorCritic:
    def __init__(self, env, actor_lr, critic_lr,
                 gamma, device, name, fc1=2048, fc2=2048):

        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.state_dim
        self.high_action = env.action_space.high

        self.actor = PolicyNet(self.state_dim, self.action_dim, self.high_action, 
                               device,fc1, fc2, name).to(device)
        self.critic = ValueNet(self.state_dim, fc1, fc2).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr) 
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        action = self.actor(state)
        action_mean = action[:, :self.action_dim] 
        action_std = action[:, self.action_dim:]
        dist = torch.distributions.Normal(action_mean, action_std)
        action_f = dist.sample()
        #action_f = torch.tanh(action_f)*torch.tensor([self.high_action]).to(self.device)
        action_f = action_f.cpu().detach().numpy()[0]
        return action_f

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions'])).view(-1, self.action_dim).to(
                               self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']),
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']),
                             dtype=torch.float).view(-1, 1).to(self.device)

        # TD target
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)  # TD error

        action = self.actor(states)
        action_mean = action[:, :self.action_dim] 
        action_std = action[:, self.action_dim:]
        dist = torch.distributions.Normal(action_mean, action_std)

        log_probs = dist.log_prob(actions)
        actor_loss = torch.mean(-log_probs * td_delta.detach())

        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def load_model(self):
        print('.... loading model ....')
        self.actor.load()