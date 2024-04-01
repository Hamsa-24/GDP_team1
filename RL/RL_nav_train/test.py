import torch
from rl_utils import initialize_plot, update_plot
import matplotlib.pyplot as plt
from environment import Environment2D, Environment3D
from actor_critic import ActorCritic
from DDPG import DDPG


num_episodes = 1000
device = torch.device("cuda") if torch.cuda.is_available() else torch.device( "cpu")

state_dim = 4
action_dim = 1

env = Environment2D(state_dim, action_dim)
agent = ActorCritic(env, 0, 0, 0, device, fc1=1024, fc2=512, 
                    name='AC2d_5c')
agent.load_model()

PLOT2d = True
PLOT3d = False
PLOT_episodes = range(num_episodes)

num_success = 0
return_list = []
for i_episode in range(num_episodes):
    if i_episode in PLOT_episodes:
        fig, ax, tmp_objects = initialize_plot(PLOT2d, PLOT3d)

    state = env.reset()
    done = False

    while not done:
        action = agent.take_action(state)
        next_state, reward, done = env.step(action)
        state = next_state
        if i_episode in PLOT_episodes:
            tmp_objects = update_plot(tmp_objects, fig, ax, env, PLOT2d, PLOT3d)

    if reward > 0:
        num_success +=1
    plt.close()

print("Success rate: {:.0%}".format(num_success / num_episodes))

