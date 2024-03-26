import torch
import torch.nn.functional as F
import numpy as np
import os


class Replay_buffer():
    def __init__(self, max_size=1000000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.batch_size = 100

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, high_action, device, fc1, fc2, name='model1'):
        super(Actor, self).__init__()

        self.l1 = torch.nn.Linear(state_dim, fc1)
        self.l2 = torch.nn.Linear(fc1, fc2)
        self.l3 = torch.nn.Linear(fc2, action_dim)

        self.high_action = high_action
        self.name = name
        self.device = device

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        #x = torch.tensor([self.high_action]).to(self.device) * torch.tanh(x)
        return x
    
    def save(self):
        torch.save(self.state_dict(), os.path.join('models/nav', self.name))

    def load(self):
        self.load_state_dict(torch.load(os.path.join('models/nav', self.name)))


class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim, fc1, fc2):
        super(Critic, self).__init__()

        self.l1 = torch.nn.Linear(state_dim + action_dim, fc1)
        self.l2 = torch.nn.Linear(fc1 , fc2)
        self.l3 = torch.nn.Linear(fc2, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, env, actor_lr, critic_lr,
                 gamma, device, name, fc1=1024, fc2=512, tau=5e-2):
        
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.state_dim
        #self.state_dim = env.observation_space.shape[0]
        self.high_action = env.action_space.high
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        self.actor = Actor(self.state_dim, self.action_dim, self.high_action,
                           device, fc1, fc2, name).to(device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.high_action, 
                                  device, fc1, fc2, name).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(self.state_dim, self.action_dim, fc1, fc2).to(device)
        self.critic_target = Critic(self.state_dim, self.action_dim, fc1, fc2).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.replay_buffer = Replay_buffer()

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def take_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):

        for it in range(1):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(self.replay_buffer.batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(1-d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * self.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def load_model(self):
        print('.... loading model ....')
        self.actor.load()