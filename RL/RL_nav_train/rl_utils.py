from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import matplotlib.pyplot as plt
from plot_utils import plot_environment3d, plot_environment2d

class ReplayBuffer:
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

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent0(env, agent, num_episodes, PLOT):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                fig = plt.figure(figsize=(10,6))
                ax = fig.add_subplot(111, projection='3d')
                ax.set_xlim(0, 12)
                ax.set_ylim(0, 12)
                ax.set_zlim(0, 12)
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                    if PLOT:
                        plot_environment3d(env.robot_position, env.robot_orientation, env.target_zone, env.forbidden_zone, ax)
                        plt.pause(0.00001)
                        fig.canvas.draw_idle()  # Mise à jour du graphique
                        #fig.canvas.flush_events()  # Vidage des événements pour le graphique

                plt.close()
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

    return return_list

def train_on_policy_agent(env, agent, num_episodes, save=False, PLOT2d=False, PLOT3d=False, PLOT_episodes=None):
    return_list = []
    j = 0
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                if j in PLOT_episodes:
                    fig, ax, tmp_objects = initialize_plot(PLOT2d, PLOT3d)

                best_return = 0
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward

                    if j in PLOT_episodes:
                        tmp_objects = update_plot(tmp_objects, fig, ax, env, PLOT2d, PLOT3d)

                return_list.append(episode_return)
                agent.update(transition_dict)

                if episode_return > best_return and save:
                    best_return = episode_return
                    agent.actor.save()

                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
                j += 1
                plt.close()

    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                

def discretize(min_value, max_value, num, x):
    res = []
    categories = np.linspace(min_value, max_value, num=num-1)

    for x0 in x.cpu().detach().numpy()[0]:
        if x0 <= min_value: res.append(0)
        if x0 > max_value: res.append(num-1)
        for i in range(len(categories[:-1])):
            if x0 > categories[i] and x0 <= categories[i+1]:
                res.append(i)
                break
    return res

def undiscretize(min_value, max_value, num, x):
    categories = np.linspace(min_value, max_value, num=num-1)
    for i in range(len(x)):
        if x[i] < num-1: x[i] = categories[x[i]] 
        else: x[i] = categories[-1]
    return x

def format_action(min_value, max_value, x): 
    for i in range(len(x)):       
        x[i] = max(x[i], min_value) if x[i] < 0 else min(x[i], max_value)
    return x

def normalize(states): #depreciated
    for i in range(len(states)):
        states[i][0] = states[i][0] / 10 #Position agent
        states[i][1] = states[i][1] / 10 #Position agent
        states[i][2] = states[i][2] / 10 #Position agent
        states[i][3] = states[i][3] / 10 #Position arrivée
        states[i][4] = states[i][4] / 10 #Position arrivée
        states[i][5] = states[i][5] / 10 #Position arrivée
        states[i][6] = states[i][6] / 10 #Distance to target
        states[i][7] = (states[i][7] - (-np.pi/12)) / (2 * np.pi/12) #Orientation theta
        states[i][8] = (states[i][8] - (-np.pi/12)) / (2 * np.pi/12) #Orientation phi
        states[i][9] = states[i][9] #Présence obstacle en face
    return states


def update_orientation(orientation, d_orientation):
        orientation_ = orientation + d_orientation
        if orientation[0] + d_orientation[0] > np.pi:
            orientation_[0] = orientation[0] + d_orientation[0] - 2*np.pi
        if orientation[0] + d_orientation[0] < -np.pi:
            orientation_[0] = 2*np.pi + orientation[0] + d_orientation[0] 
        if orientation[1] + d_orientation[1] > np.pi/2:
            orientation_[1] = np.pi - orientation[1] - d_orientation[1]
        if orientation[1] + d_orientation[1] < -np.pi/2:
            orientation_[1] = -np.pi - orientation[1] - d_orientation[1]

        return orientation_


def initialize_plot(PLOT2d, PLOT3d):
    if PLOT2d:
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        ax.set_xlim(-1, 12)
        ax.set_ylim(-1, 12)
        text1, text2, text3, line1, line2 = ax.text(-100,-100,''), ax.text(-100,-100,''), ax.text(-100,-100,''), \
                                            ax.text(-100,-100,''), ax.text(-100,-100,'')
        tmp_objects = [text1, text2, text3, line1, line2]
        return fig, ax, tmp_objects
    elif PLOT3d:
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1, 12)
        ax.set_ylim(-1, 12)
        ax.set_zlim(-1, 12)
        text1, text2, text3, lines = ax.text(-100,-100,-100,''), ax.text(-100,-100,-100,''), ax.text(-100,-100,-100,''), \
                                    [ax.text(-100,-100,-100,'') for i in range(4)]
        tmp_objects = [text1, text2, text3]
        for line in lines:
             tmp_objects.append(line)
        return fig, ax, tmp_objects
    return None, None, None
    

def update_plot(objects, fig, ax, env, PLOT2d, PLOT3d):
    if PLOT2d:
        for object in objects:
            object.remove()
        line1, line2 = plot_environment2d(env.count, env.robot_position, env.robot_orientation, env.target_zone, env.forbidden_zone, ax)
        text1=ax.scatter(-100, -100, label=f"Angle: {env.angle_to_target()*180/np.pi:.0f}°", c='r')
        text2=ax.scatter(-100, -100, label=f"Distance: {env.dist_to_target():.1f}", c='r')
        text3=ax.scatter(-100, -100, label=f"Orientation: {env.robot_orientation[0]*180/np.pi:.0f}°", c='r')
        plt.legend()
        plt.pause(0.00001)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        tmp_objects = [text1, text2, text3, line1, line2]
        return tmp_objects
    
    if PLOT3d:
        for object in objects:
            object.remove()
        lines = plot_environment3d(env.count, env.robot_position, env.robot_orientation, env.target_zone, env.forbidden_zone, ax)
        text1=ax.scatter(-100, -100, -100, label=f"Angle: {env.angle_to_target()*180/np.pi:.0f}°", c='r')
        text2=ax.scatter(-100, -100, -100, label=f"Distance: {env.dist_to_target():.1f}", c='r')
        text3=ax.scatter(-100, -100, -100,label=f"Orientation: {env.robot_orientation[0]*180/np.pi:.0f}°", c='r')
        plt.legend()
        plt.pause(0.00001)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        tmp_objects = [text1, text2, text3]
        for line in lines:
            tmp_objects.append(line)
        return tmp_objects