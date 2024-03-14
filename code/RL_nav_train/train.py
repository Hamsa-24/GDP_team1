import torch
from rl_utils import train_on_policy_agent
from plot_utils import moving_average
import matplotlib.pyplot as plt
from environment import Environment2D, Environment3D
from actor_critic import ActorCritic



actor_lr = 1e-5 #5
critic_lr = 1e-6 #6
num_episodes = 25000
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device( "cpu")

state_dim = 3
action_dim = 1

env = Environment2D(state_dim, action_dim)
agent = ActorCritic(env, actor_lr, critic_lr, gamma, device)

PLOT2d = True
PLOT3d = False
return_list = train_on_policy_agent(env, agent, num_episodes, 
                                    PLOT2d=PLOT2d,
                                    PLOT3d=PLOT3d)


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
