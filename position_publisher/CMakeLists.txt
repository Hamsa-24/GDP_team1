cmake_minimum_required(VERSION 3.8)
project(position_publisher)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclpy REQUIRED)


install(PROGRAMS  
  position_subscriber.py
  image_subscriber.py
  position_publisher_test.py 
  reset_simulation.py 
  RL_communication.py
  env.py
  reset_simulation_server.py
  range_subscriber.py
  pos_range_cogload.py
  DESTINATION lib/${PROJECT_NAME}/src )

ament_package()







