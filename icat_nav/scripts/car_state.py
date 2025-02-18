#!/usr/bin/env python
import rospy
import numpy as np
import tf
from geometry_msgs.msg import Twist
from collections import deque

class CarStateNode:
    def __init__(self):
        rospy.init_node('car_state_node')
        
        self.state_estimate = np.zeros(6)  # [x, y, theta, vx, vy, omega]
        self.covariance_matrix = np.diag([0.1, 0.1, np.radians(5), 0.1, 0.1, np.radians(5)])
        self.state_buffer = deque(maxlen=100)  # Buffer for past states
        self.tf_listener = tf.TransformListener()
        self.robot_frames = ["st", "a4", "a2", "a6"]
        self.robot_frames = ["st"]
        self.car_state_pubs = {
            robot: rospy.Publisher(f'/{robot}/car_state', Twist, queue_size=10)
            for robot in self.robot_frames
        }

    def make_twist(self, car_state):
        x, y, theta = car_state
        cmd = Twist()
        cmd.linear.x = x
        cmd.linear.y = y
        cmd.linear.z = theta
        cmd.angular.x = 0
        cmd.angular.y = 0
        cmd.angular.z = 0
        return cmd

    def get_robot_state(self, robot_name):
        try:
            self.tf_listener.waitForTransform("/map", f"/{robot_name}/base_link", rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform("/map", f"/{robot_name}/base_link", rospy.Time(0))
            _, _, theta = tf.transformations.euler_from_quaternion(rot)
            return trans[0], trans[1], theta
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn(f"Could not get TF for {robot_name}")
            return None

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            for robot in self.robot_frames:
                state = self.get_robot_state(robot)
                if state is not None:
                    twist_state = self.make_twist(state)
                    self.car_state_pubs[robot].publish(twist_state)
            rate.sleep()

if __name__ == '__main__':
    try:
        node = CarStateNode()
        node.run()
    except rospy.ROSInterruptException:
        pass