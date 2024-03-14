import torch
import torch.nn.functional as F


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 2*action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        middle = int(len(x[0,:])/2)
        return torch.cat((x[:,:middle], torch.nn.functional.softplus(x[:,middle:])+1e-5), dim=1)
    

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class ActorCritic:
    def __init__(self, env, actor_lr, critic_lr,
                 gamma, device):

        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.state_dim
        self.high_action = env.action_space.high

        self.actor = PolicyNet(self.state_dim, self.action_dim).to(device)
        self.critic = ValueNet(self.state_dim).to(device)  

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr) 
        
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state)
        action_mean = action[:, :self.action_dim] 
        action_std = action[:, self.action_dim:]
        dist = torch.distributions.Normal(action_mean, action_std)
        sampled_actions = dist.sample()
        action_f = torch.tanh(sampled_actions)*torch.tensor(self.high_action).to(self.device)
        action_f = sampled_actions.cpu().detach().numpy()[0]
        return action_f

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