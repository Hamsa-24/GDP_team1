import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plot_utils import plot_environment, save_rewards



# Définition de l'environnement et du robot
class Environment:
    def __init__(self):
        self.robot_orientation = np.array([np.random.uniform(-np.pi, np.pi),
                                           np.random.uniform(-np.pi/2, np.pi/2)])
        self.robot_position = np.zeros(3)  # position initiale du robot
        self.robot_size = np.full((1,3), 0.1)  # taille du côté du cube représentant le robot
        self.target_zone = np.array([[9, 9, 9], [10, 10, 10]])# position cible
        self.forbidden_zone = np.array([[4.5, 4.5, 4.5], [6.5, 6.5, 6.5]])  # coins de la zone interdite

    # réinitialisation de la position et de la direction du robot
    def reset(self):
        self.robot_position = np.zeros(3)  
        self.robot_orientation = np.array([np.random.uniform(-np.pi, np.pi),
                                         np.random.uniform(-np.pi/2, np.pi/2)])
        return self.get_state()
    
    def dist_to_target(self):
        return np.linalg.norm(self.robot_position - self.target_zone.mean(axis=0))
    
    def obstacle_in_field_of_vision(self, angle_vision=np.pi/12):
        # Convertir l'orientation en vecteur directionnel
        position = self.robot_position
        orientation = self.robot_orientation
        position_obstacle = self.forbidden_zone
        direction = np.array([np.cos(orientation[0]) * np.sin(orientation[1]),
                              np.sin(orientation[0]) * np.cos(orientation[1]),
                              np.sin(orientation[1])])

        # Calculer le vecteur entre la position de l'agent et le centre de l'obstacle
        vector_agent_obstacle = (position_obstacle[0] + position_obstacle[1]) / 2 - position

        # Calculer la distance entre l'agent et le centre de l'obstacle
        distance_agent_obstacle = np.linalg.norm(vector_agent_obstacle)

        # Vérifier si l'obstacle est dans le champ de vision de l'agent
        if distance_agent_obstacle > 0:
            angle_obstacle = np.arccos(np.dot(direction, vector_agent_obstacle) / distance_agent_obstacle)
            if angle_obstacle < angle_vision:
                return 1
        return 0

    def step(self, d_orientation, nstep, step_crash):
        # Mise à jour de l'orientation du robot
        d_orientation[0] = max(d_orientation[0],-np.pi/12) if d_orientation[0] < 0 else min(d_orientation[0], np.pi/12)
        d_orientation[1] = max(d_orientation[1],-np.pi/12) if d_orientation[1] < 0 else min(d_orientation[1], np.pi/12)

        self.robot_orientation += d_orientation
        # Mise à jour de la position du robot
        self.robot_position += 1/5 * np.array([np.cos(self.robot_orientation[0]) * np.sin(self.robot_orientation[1]), 
                                               np.sin(self.robot_orientation[0]) * np.cos(self.robot_orientation[1]), 
                                               np.sin(self.robot_orientation[1])])

        # Vérifie si le robot est entré en contact avec la zone interdite
        if self.check_collision(self.robot_position):
            return self.get_state(), -100, True  # collision avec la zone interdite, fin de la simulation
        
        # Vérifie si le robot a atteint la position cible
        if self.check_target_reached(self.robot_position):
            return self.get_state(), 100, True  # robot atteint la position cible, fin de la simulation
        
        if self.is_crashed(nstep, step_crash):
            return self.get_state(), -self.dist_to_target()*5, True # robot crashé, fin de la simulation

        return self.get_state(), 0, False

    def check_collision(self, position):
        # Vérifie si le robot est en collision avec la zone interdite
        robot_corners = self.get_robot_corners(position)
        for corner in robot_corners:
            if np.all(corner >= self.forbidden_zone[0]) and \
               np.all(corner <= self.forbidden_zone[1]):
                return True
        return False

    def get_robot_corners(self, position):
        # Retourne les coins du cube représentant le robot
        half_size = self.robot_size / 2
        robot_corners = np.array([
            [position[0] - half_size, position[1] - half_size, position[2] - half_size],
            [position[0] - half_size, position[1] - half_size, position[2] + half_size],
            [position[0] - half_size, position[1] + half_size, position[2] - half_size],
            [position[0] - half_size, position[1] + half_size, position[2] + half_size],
            [position[0] + half_size, position[1] - half_size, position[2] - half_size],
            [position[0] + half_size, position[1] - half_size, position[2] + half_size],
            [position[0] + half_size, position[1] + half_size, position[2] - half_size],
            [position[0] + half_size, position[1] + half_size, position[2] + half_size]
        ])
        return robot_corners

    def check_target_reached(self, position):
    
        # Calcule les coordonnées du coin inférieur et du coin supérieur du cube du drone
        drone_lower_corner = position - self.robot_size / 2
        drone_upper_corner = position + self.robot_size / 2
    
        # Vérifie si tous les coins du cube du drone sont contenus dans la zone en forme de pavé droit
        return (np.all(drone_lower_corner >= self.target_zone[0]) and 
                np.all(drone_upper_corner <= self.target_zone[1]))
    
    def is_crashed(self, nstep, step_crash):
        #return np.random.rand() < prob_crash(nstep)
        return nstep == step_crash

    def get_state(self):
        # Retourne l'état actuel de l'environnement
        distance_to_obstacle = np.linalg.norm(self.robot_position - self.forbidden_zone.mean(axis=0))
        return np.concatenate((self.robot_position, 
                              (self.target_zone[0]+self.target_zone[1])/2, 
                              [self.dist_to_target()],
                               self.robot_orientation,
                              [self.obstacle_in_field_of_vision()]))

# Définition du réseau neuronal
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Paramètres d'apprentissage
TRAIN = True
PLOT = False
gamma = 0.99  # facteur d'actualisation
epsilon = 0.0  # exploration initiale #1.0
epsilon_decay = 1.0  # taux de décroissance de l'exploration #0.95
epsilon_min = 0.0  # exploration minimale #0.01
learning_rate = 2e-3  # taux d'apprentissage

# Initialisation de l'environnement et du NN
env = Environment()
input_size = 10  # dimensions de l'observation (position du robot, position de l'arrivée, distance à l'arrivée, cap, distance obstacle devant
output_size = 2  # dimensions de l'action (Directions d_theta et d_phi)
model = QNetwork(input_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Algorithme d'apprentissage

if TRAIN:
    num_episodes = 15000
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        total_reward = 0
        nstep = 0
        step_crash = int(np.random.normal(100, 10))

        # Initialisation de la figure
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 12)
        ax.set_zlim(0, 12)

        while not done:
            # Choix de l'action
            if np.random.rand() < epsilon: #depreciated, voir valeurs de epsilon
                theta = np.random.uniform(-np.pi, np.pi)
                phi = np.random.uniform(-np.pi/2, np.pi/2)
                action = np.array([np.cos(theta), np.sin(theta), np.sin(phi)])/5
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action = q_values.cpu().numpy()[0]

            # Exécution de l'action dans l'environnement
            next_state, reward, done = env.step(action, nstep, step_crash)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            total_reward += reward

            # Calcul de la target Q-value
            with torch.no_grad():
                target = reward + gamma * torch.max(model(next_state))

            # Calcul de la Q-value prédite
            q_value = model(state)[0][torch.argmax(torch.tensor(action))]

            # Calcul de la loss
            loss = criterion(q_value, target)

            # Optimisation du réseau neuronal
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            nstep += 1

            if PLOT:
                plot_environment(env.robot_position, env.robot_orientation, env.target_zone, env.forbidden_zone, ax)
                plt.pause(0.00001)
                fig.canvas.draw_idle()  # Mise à jour du graphique
                #fig.canvas.flush_events()  # Vidage des événements pour le graphique

        plt.close()

        # Mise à jour de l'exploration
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
        rewards.append(total_reward)

    save_rewards(rewards, 'test')

# Une fois l'apprentissage terminé, l'agent peut être utilisé pour naviguer dans l'environnement

#fig = plt.figure(figsize=(10,6))
#ax = fig.add_subplot(111, projection='3d')
#plot_environment(env.robot_position, env.target_zone, env.forbidden_zone)
#plt.show()