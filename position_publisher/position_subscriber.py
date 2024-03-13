#!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Pose

# class PositionPublisher(Node):
#     def __init__(self):
#         super().__init__('position_publisher')
#         self.subscription = self.create_subscription(
#             Pose,
#             '/simple_robot/tf',  # Change this topic to the actual topic name
#             self.position_callback,
#             10)
#         self.subscription  # Prevent unused variable warning

#     def position_callback(self, msg):
#         # Process the received position message
#         position = msg.position
#         print("Robot Position:")
#         print("x:", position.x)
#         print("y:", position.y)
#         print("z:", position.z)

# def main(args=None):
#     rclpy.init(args=args)
#     position_subscriber = PositionPublisher()

#     def timer_callback():
#         # You can create a dummy message here or handle the absence of message accordingly
#         msg = None  # Placeholder for the received message
#         position_subscriber.position_callback(msg)

#     timer_period = 1.0  # seconds
#     timer = position_subscriber.create_timer(timer_period, timer_callback)

#     rclpy.spin(position_subscriber)
#     position_subscriber.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

import rclpy
from rclpy.node import Node
# from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
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
            10)  # Set QoS depth to 10000
        self.timer = self.create_timer(1, self.save_to_csv)  # Timer to save to CSV every 1 second
        self.csv_file = 'simple_drone_positions.csv'
        self.latest_position = None
        self.latest_rotation = None
        # self.positions = []


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

    def save_to_csv(self):
        # # Save positions to CSV file
        # with open(self.csv_file, mode='a') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(self.positions)
        # self.positions = []  # Clear positions list for next cycle
        if self.latest_position is not None:
            if env.is_position_allowed([self.latest_position.x, self.latest_position.y, self.latest_position.z]) == True:
            
                # Save the latest position to CSV file
                with open(self.csv_file, mode='a') as file:
                    writer = csv.writer(file)
                    writer.writerow([self.latest_position.x, self.latest_position.y, self.latest_position.z,self.latest_rotation,"Flying"])
                # Clear the latest position
                self.latest_position = None
                self.latest_rotation = None
            else : 
                                # Save the latest position to CSV file
                with open(self.csv_file, mode='a') as file:
                    writer = csv.writer(file)
                    writer.writerow([self.latest_position.x, self.latest_position.y, self.latest_position.z,self.latest_rotation,"You Crashed"])
                    
                    reset_world()
def main(args=None):
    rclpy.init(args=args)
    tf_listener = TFListener()
    rclpy.spin(tf_listener)
    tf_listener.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()