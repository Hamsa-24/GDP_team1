#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/simple_drone/front/image_raw',
            self.image_callback,
            10)
        self.cv_bridge = CvBridge()

    def image_callback(self, msg):
        # Convertir le message d'image ROS en image OpenCV
        image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Afficher l'image
        cv2.imshow('Image from simple_drone/front/image_raw', image)
        cv2.waitKey(1)  # Attendre une petite période pour permettre l'affichage de l'image

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()