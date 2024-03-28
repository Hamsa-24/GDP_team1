files : 

setup.py : file that is emptu and required to cretae executables in a ros2 package

reset_simulation.py : file to try out the service ros2 to reset the gazebo simulation

range_subscriber.py : file to subscribe to the sonar topic facing forward and print the value every second in a csv file 

position_subscriber.py : file to susbcribe to sonar/odometry topics and print data + status of the flight accordiing to allowed positions in env.

position_publisher_test.py : same as position_subscriber.py without the sonar topic and with a script to get the CogLoad from arduino serial monitor.
                            To be associated with serial_plotter.py in order to work

pos_range_cogload.py : fusion f every functionnalities to be used for learning.