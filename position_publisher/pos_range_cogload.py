#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
# from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
import csv
from env import Environment3D
from reset_simulation_server import reset_world

size_apartment = [20.08,22.19,14.79]
size_fast_food = [24.91, 15.725,11.015]
size_gas_station = [17.52,25.53,7.675]
size_house_1 = [15.50,16.39,7.68]
size_house_2 = [12.48,8.94,7.19]
size_house_3 = [4.57,11.79,10.61]
size_law_office = [6.84,5.43,13.92]
size_osrf_first_office = [26.18,18.30,5.73]
size_radio_tower = [11.72,13.39,44.19]
size_salon = [7.21,5.37,11.38]
size_thrift_shop = [7.21,5.43,11.38]
size_post_office = [10.40,7.30,3.95]

#Creating the environment
env = Environment3D((100, 100, 100))  # Create a 3D environment of size 100x100x100
env.create_building([1.573,27.565,0.0],0,size_apartment)
env.create_building([87.36,-21.48,0.0],1.57,size_apartment)
env.create_building([28.25,-13.93,0.0],0,size_fast_food)
env.create_building([61.88,-30.82,0],0,size_gas_station)
env.create_building([-27.49,25.344,0.0],1.57,size_house_1)
env.create_building([61.74,22.76,0.0],1.57,size_house_1)
env.create_building([86.62,27.93,0.0],0,size_house_1)
env.create_building([76.71,13.4,0.0],0,size_house_2)
env.create_building([1.61,-14.15,0.0],0,size_house_2)
env.create_building([-0.59,-26.6,0.0],1.57,size_house_2)
env.create_building([96.43,14.14,0.0],0,size_house_3)
env.create_building([-33.39,-21.3,0.0],1.57,size_house_3)
env.create_building([-28.126,-32.93,0.0],0,size_house_3)
env.create_building([-25.739,-19.38,0.0],1.57,size_house_3)
env.create_building([8.47,9.356,0.0],0,size_law_office)
env.create_building([-34.886,-9.408,0.0],0,size_law_office)
env.create_building([27.003,20.15,0.0],1.57,size_osrf_first_office)
env.create_building([14.0,-22.5,1.57965648],0,size_radio_tower)
env.create_building([15.7,9.356,0.0],0,size_salon)
env.create_building([-27.65,-9.35,0.0],0,size_salon)
env.create_building([1.35,9.356,0.0],0,size_thrift_shop)
env.create_building([-26.686,11.119,0.0],0,size_post_office)

class TFListener(Node):
    def __init__(self):
        super().__init__('tf_listener')
        self.subscription = self.create_subscription(
            Odometry,
            '/simple_drone/odom',
            self.tf_callback,
            10)
        
        # Abonnement aux données du sonar
        self.sonar_subscription = self.create_subscription(
            Range,
            '/simple_drone/sonar/out',
            self.sonar_callback,
            10) 
        self.csv_file = 'simple_drone_positions.csv'
        self.latest_position = None
        self.latest_rotation = None
        self.latest_range = None
        self.arduino_data = None

    def tf_callback(self, msg : Odometry):
        # # Extract position information from the TransformStamped message
        # position = msg.position
        # print(position)
        # self.positions.append([position.x, position.y, position.z])
        position_x = msg.pose.pose.position.x
        position_y = msg.pose.pose.position.y
        position_z = msg.pose.pose.position.z

        print("Robot Position:")
        print("x:", position_x)
        print("y:", position_y)
        print("z:", position_z)

        # # Add positions to list
        # self.positions.append([position_x, position_y, position_z])
        self.latest_position = msg.pose.pose.position
        self.latest_rotation = msg.pose.pose.orientation.z

    def sonar_callback(self, msg : Range):
        range = msg.range

        print("distance from obstacle in front of robot:")
        print("d:", range)
 

        # # Add positions to list
        # self.positions.append([position_x, position_y, position_z])
        self.latest_range = range
        self.save_to_csv()
    
    def bpm_callback(self):
        with open("/home/blechardoy/Cranfield/Python/Deep_learning/RL_Laboratory/serial_data.csv", mode='r') as file:
            last_line = None
            for line in file:
                last_line = line

        if last_line is not None:
            values = last_line.strip().split(',')
            self.arduino_data = values[-1]

    def save_to_csv(self):
        self.bpm_callback()
        if self.latest_position is not None and self.arduino_data is not None:
            with open(self.csv_file, mode='a') as file:
                if env.is_position_allowed([self.latest_position.x, self.latest_position.y, self.latest_position.z]):
                    writer = csv.writer(file)
                    writer.writerow([self.latest_position.x, self.latest_position.y, self.latest_position.z, self.latest_rotation,self.latest_range, self.arduino_data, "Flying"])
                elif env.is_landed([self.latest_position.x, self.latest_position.y, self.latest_position.z]):
                    writer = csv.writer(file)
                    writer.writerow([self.latest_position.x, self.latest_position.y, self.latest_position.z, self.latest_rotation, self.latest_range, self.arduino_data, "You landed successfully"])
                    reset_world()
                else:
                    writer = csv.writer(file)
                    writer.writerow([self.latest_position.x, self.latest_position.y, self.latest_position.z, self.latest_rotation, self.latest_range, self.arduino_data, "You Crashed"])
                    reset_world()

    def update_arduino_data(self, arduino_data):
        self.arduino_data = arduino_data
        self.save_to_csv()  # Call save_to_csv after receiving arduino data

def main(args=None):
    rclpy.init(args=args)
    tf_listener = TFListener()
    rclpy.spin(tf_listener)
    tf_listener.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()