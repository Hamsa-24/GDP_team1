import rclpy
from rclpy.node import Node
# from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import csv
from env import Environment3D
from reset_simulation_server import reset_world   

def append_from_csv(input_csv_file,output_csv_file):
    # Read the last line of the input CSV file
    with open(input_csv_file, mode='r') as file:
        last_line = None
        for line in file:
            last_line = line

    # Extraire la dernière valeur de last_line
    if last_line is not None:
        # Diviser la ligne en une liste de valeurs
        values = last_line.strip().split(',')
        # Extraire la dernière valeur
        last_value = values[-1]

        # Ajouter la dernière valeur à la fin de chaque ligne du fichier CSV de sortie
        with open(output_csv_file, mode='a') as output_file:
            output_file.write(',' + last_value )  # Concaténer les éléments existants et la dernière valeur, puis ajouter un saut de ligne

def main(args=None):
    append_from_csv("/home/blechardoy/Cranfield/Python/Deep_learning/RL_Laboratory/serial_data.csv","test_csv.csv")

if __name__ == '__main__':
    main()