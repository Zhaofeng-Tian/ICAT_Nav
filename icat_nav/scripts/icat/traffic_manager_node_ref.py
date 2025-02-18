#!/usr/bin/env python3
import rospy
import rospkg
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from nav_gym.obj.geometry.util import topi
import random
from topo import *
from car import build_car, get_car_param
from scipy.spatial.transform import Rotation as R
import time
from traffic_manager import TrafficManager

if_sim = False

class TrafficManagerNode:
    def __init__(self):
        rospy.init_node('traffic_manager_node')

        # Define robot names
        self.robots = [f"robot{i+1}" for i in range(4)]  # Adjust for more robots

        # Paths and Parameters
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('icat_nav')
        self.wpt_dist = 0.05
        self.dt = 0.1
        self.n_car = len(self.robots)  # Set number of cars dynamically
        self.n_node = 42
        self.car_param = get_car_param()
        self.car_info = {"hl": 0.1775, "hw": 0.10, "amax": 0.3, "amin": -0.3, "jerkmax": 1.0}
        self.car_states = [np.zeros(5) for _ in range(self.n_car)]

        # Load map data
        self.node_list = load_edges(package_path + '/data/carto_icat_nodes.json')
        self.edge_list = load_edges(package_path + '/data/carto_icat_edges.json')
        self.G = build_graph(self.node_list, self.edge_list)
        self.start_nodes, self.goal_nodes = self.sample_sg_nodes()

        # Traffic Manager
        self.TM = TrafficManager(
            node_list=self.node_list, edge_list=self.edge_list, G=self.G, 
            n_car=self.n_car, car_states=self.car_states, car_info=self.car_info,
            start_nodes=self.start_nodes, goal_nodes=self.goal_nodes, wpts_dist=self.wpt_dist
        )

        # ROS Subscribers and Publishers
        self.subs = []
        self.cmd_pubs = {}
        self.traj_pubs = {}

        for i, robot in enumerate(self.robots):
            self.subs.append(rospy.Subscriber(f'{robot}/car_state', Twist, self.car_state_callback, callback_args=i))
            self.subs.append(rospy.Subscriber(f'{robot}/vel_raw', Twist, self.vel_raw_callback, callback_args=i))
            self.cmd_pubs[robot] = rospy.Publisher(f'{robot}/cmd_vel', Twist, queue_size=10)
            self.traj_pubs[robot] = rospy.Publisher(f'{robot}/traj_path', Path, queue_size=10)

    def sample_sg_nodes(self):
        node_arr = list(range(1, self.n_node + 1))
        nodes = random.sample(node_arr, 2 * self.n_car)
        return nodes[:self.n_car], nodes[self.n_car:]

    def car_state_callback(self, msg, idx):
        """ Updates car state with position and yaw """
        x, y, theta = msg.linear.x, msg.linear.y, msg.linear.z
        self.car_states[idx][:3] = np.array([x, y, theta])

    def vel_raw_callback(self, msg, idx):
        """ Updates car state with velocity and yaw rate """
        vx, vy, omega = msg.linear.x, msg.linear.y, msg.linear.z
        self.car_states[idx][3:] = np.array([np.hypot(vx, vy), omega])

    def make_traj_msg(self, traj):
        """ Converts trajectory array to a ROS Path message """
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = rospy.Time.now()
        for point in traj:
            traj_pose = PoseStamped()
            traj_pose.header.frame_id = "map"
            traj_pose.header.stamp = rospy.Time.now()
            traj_pose.pose.position.x, traj_pose.pose.position.y = point[0], point[1]
            q = R.from_euler('zyx', [point[2], 0, 0]).as_quat()
            traj_pose.pose.orientation.x, traj_pose.pose.orientation.y = q[0], q[1]
            traj_pose.pose.orientation.z, traj_pose.pose.orientation.w = q[2], q[3]
            path.poses.append(traj_pose)
        return path

    def make_cmd_vel(self, raw_cmd):
        """ Creates a Twist command from velocity inputs """
        v, w = raw_cmd
        cmd = Twist()
        cmd.linear.x, cmd.linear.y = v, w
        return cmd

    def compute_control(self, H, T, S):
        """ Computes velocity commands based on trajectory """
        Sg = T[H]
        ds = np.hypot(Sg[0] - S[0], Sg[1] - S[1])
        v = ds / (H * self.dt)
        w = topi(Sg[2] - S[2]) / (H * self.dt)
        return [v, w]

    def run(self):
        """ Main loop that updates traffic state, computes control, and publishes commands """
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            start_time = time.time()

            # Normalize theta values
            for i in range(self.n_car):
                self.car_states[i][2] = topi(self.car_states[i][2])

            self.TM.traffic_state_update(self.car_states)
            trajbuffer = self.TM.get_traj_buffer()
            statebuffer = self.TM.get_state_buffer()

            cmd_list = []
            for i, robot in enumerate(self.robots):
                traj_path = self.make_traj_msg(trajbuffer[i])
                self.traj_pubs[robot].publish(traj_path)

                cmd = self.compute_control(min(2, len(trajbuffer[i])), trajbuffer[i].copy(), self.car_states[i].copy())
                cmd = [min(0.2, cmd[0]), cmd[1] * 0.045]  # Limit velocity
                rospy.loginfo(f"{robot} command: {cmd}")
                cmd_list.append(cmd)

            if not if_sim:
                for i, robot in enumerate(self.robots):
                    self.cmd_pubs[robot].publish(self.make_cmd_vel(cmd_list[i]))

            rospy.loginfo(f"Cycle Time: {time.time() - start_time:.4f}s")
            rate.sleep()

if __name__ == '__main__':
    try:
        node = TrafficManagerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
