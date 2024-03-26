import torch
from rl_utils import train_on_policy_agentDDPG, train_on_policy_agentAC
from plot_utils import moving_average
import matplotlib.pyplot as plt
from environment import Environment2D, Environment3D
from actor_critic import ActorCritic
from DDPG import DDPG


actor_lr = 1e-5 #5
critic_lr = 1e-10#6
num_episodes = 100000
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device( "cpu")

state_dim = 4
action_dim = 1

env = Environment2D(state_dim, action_dim)
agent = ActorCritic(env, actor_lr, critic_lr, gamma, device, 
             name='AC2d_4', fc1=1024, fc2=512)

PLOT2d = True
PLOT3d = False
PLOT_episodes = [1000, 5000, 20000, 50000, 99996, 99997, 99998, 99999]
return_list = train_on_policy_agentAC(env, agent, num_episodes, save=True,
                                        PLOT2d=PLOT2d,
                                        PLOT3d=PLOT3d,
                                        PLOT_episodes=PLOT_episodes)


episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic')
plt.show()

mv_return = moving_average(return_list, 299)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic')
plt.show()
