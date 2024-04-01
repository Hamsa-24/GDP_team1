import torch 
import  numpy as np
import matplotlib.pyplot as plt
from DQN import ReplayBuffer, DQN
from environment import Environment3D
from plot_utils import moving_average
from nav_env import NavEnvironment
from actor_critic import ActorCritic
import pygame
# from geometry import set_projection


lr = 2e-3
num_episodes = 500
n_sample_paths = 30 # Number of different paths generated to find best path for the next instruction
n_actions_per_step = 15 # Number of next actions calculated with the nav model for next instruction derivation
safe_altitude = 5 # Travel altitude, above target roof altitude # temporary ?
gamma = 0.98
epsilon = 0.01
target_update = 3
buffer_size = 1000
minimal_size = 32
batch_size = 16
SAVE = False
LOAD = False
TEST = False
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

replay_buffer = ReplayBuffer(buffer_size)

state_dim = 10
action_dim = 5

env = Environment3D(state_dim, action_dim, safe_altitude, 
                    n_sample_paths, n_actions_per_step)
nav_env = NavEnvironment(state_dim=4, action_dim=1, env=env)

agent = DQN(env, lr, gamma, epsilon, target_update, device, name='DQN_1')
if LOAD:
    agent.load_model()
nav_agent = ActorCritic(nav_env, 0, 0, 0, device, fc1=1024, fc2=512, 
                        name='AC2d_5b')
nav_agent.load_model()


return_list = []
best_return = 0
for i_episode in range(num_episodes):
    pygame.init()
    if TEST:
        window = pygame.display.set_mode((1200, 800), pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)
        clock = pygame.time.Clock()
        # set_projection(*window.get_size())
    else:
        window = pygame.display.set_mode((300, 200))
        font = pygame.font.Font(None, 36)

    episode_return = 0
    state = env.reset(nav_env, nav_agent)
    _ = nav_env.reset(env=env)
    done = False
    while not done:
        action = agent.take_action(state)
        next_state, reward, done = env.step(action, nav_env, nav_agent, window)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_return += reward
                
        if replay_buffer.size() > minimal_size:
            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
            transition_dict = {
                'states': b_s,
                'actions': b_a,
                'next_states': b_ns,
                'rewards': b_r,
                'dones': b_d }
            agent.update(transition_dict)
    if episode_return > best_return and SAVE:
        best_return = episode_return
        agent.q_net.save()
        agent.target_q_net.save()
    return_list.append(episode_return)
    pygame.quit()
            
    print(f"Episode: {i_episode:.0f}, return: {episode_return:.1f}, average return: {np.mean(return_list[-10:]):.1f}")


episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN')
plt.show()

mv_return = moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN')
plt.show()