#!/usr/bin/env python3
import rospy
import rospkg
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
# from nav_gym.obj.geometry.util import topi
from collections import deque
import networkx as nx
import random
from topo import *
# from car import build_car,get_car_param
from math import pi, sin, cos, atan2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import bisect
from quintic import quintic_1d_plan
from plot import *
from traffic_manager import TrafficManager
from scipy.spatial.transform import Rotation as R
import cvxpy as cp
import time

if_sim = False

rospack = rospkg.RosPack()
package_path = rospack.get_path('icat_nav')  # Replace with your actual package name
# file_path = package_path + '/icat_nodes.json'
def topi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class TrafficManagerNode:
    def __init__(self):
        rospy.init_node('traffic_manager_node')

        self.wpt_dist = 0.05
        self.dt = 0.1

        self.car_names = ["st"]
        self.n_car = len(self.car_names)
        self.n_node = 42
        # self.car_param = get_car_param()
        self.car_info = {"hl":0.1775, "hw": 0.10, "amax":0.3, "amin":-0.3, "jerkmax": 1.0}
        self.car_states = {car: np.zeros(5) for car in self.car_names}                          
        self.Mstates = np.array([self.car_states[car] for car in self.car_names])

        self.node_list = load_edges(package_path + '/data/carto_icat_nodes.json')
        self.edge_list = load_edges(package_path + '/data/carto_icat_edges.json')
        self.G = build_graph(self.node_list, self.edge_list)
        self.start_nodes, self.goal_nodes = self.sample_sg_nodes() 
        # only for test
        self.start_nodes = [4,5,9,29, 39,38][:self.n_car]
        self.goal_nodes = [13,20,20,17, 16, 40][:self.n_car]

        self.car_state_subs = {car_name: 
                               rospy.Subscriber(f'/{car_name}/car_state',Twist, 
                                                self.car_state_callback, callback_args=car_name) 
                               for car_name in self.car_names}
        self.vel_raw_subs = {car_name:
                             rospy.Subscriber(f'/{car_name}/vel_raw', Twist, 
                                                       self.vel_raw_callback, callback_args=car_name)
                             for car_name in self.car_names}

        # # rospy.Subscriber('/ndt_pose', PoseStamped, self.ndt_pose_callback)
        # rospy.Subscriber('robot1/car_state',Twist, self.car_state_callback1)
        # rospy.Subscriber('robot1/vel_raw', Twist, self.vel_raw_callback1)
        # rospy.Subscriber('robot2/car_state',Twist, self.car_state_callback2)
        # rospy.Subscriber('robot2/vel_raw', Twist, self.vel_raw_callback2)
        # rospy.Subscriber('robot3/car_state',Twist, self.car_state_callback3)
        # rospy.Subscriber('robot3/vel_raw', Twist, self.vel_raw_callback3)
        # rospy.Subscriber('st/car_state',Twist, self.car_state_callback4)
        # rospy.Subscriber('st/vel_raw', Twist, self.vel_raw_callback4)
        self.traj_path_pubs = {car_name: rospy.Publisher(f'{car_name}/traj_path',Path, queue_size=10) for car_name in self.car_names}
        self.cmd_pubs = {car_name: rospy.Publisher(f'{car_name}/cmd_vel',Twist,queue_size=10) for car_name in self.car_names }
        # self.traj_path_pub1 = rospy.Publisher('robot1/traj_path',Path, queue_size=10)
        # self.cmd_pub1 = rospy.Publisher('robot1/cmd_vel',Twist,queue_size=10)
        # self.traj_path_pub2 = rospy.Publisher('robot2/traj_path',Path, queue_size=10)
        # self.cmd_pub2 = rospy.Publisher('robot2/cmd_vel',Twist,queue_size=10)
        # self.traj_path_pub3 = rospy.Publisher('robot3/traj_path',Path, queue_size=10)
        # self.cmd_pub3 = rospy.Publisher('robot3/cmd_vel',Twist,queue_size=10)
        # self.traj_path_pub4 = rospy.Publisher('robot4/traj_path',Path, queue_size=10)
        # self.cmd_pub4 = rospy.Publisher('robot4/cmd_vel',Twist,queue_size=10)



        self.TM = TrafficManager(node_list=self.node_list, edge_list=self.edge_list, 
                            G = self.G, n_car = self.n_car,
                            car_states=self.Mstates, car_info = self.car_info, 
                            start_nodes=self.start_nodes, goal_nodes=self.goal_nodes,
                            wpts_dist=self.wpt_dist)        



    def car_state_callback(self, msg, car_name):
        x = msg.linear.x
        y = msg.linear.y
        theta = msg.linear.z
        self.car_states[car_name][0:3] = np.array([x, y, theta])

    def vel_raw_callback(self, msg, car_name):
        v = msg.linear.x
        phi = msg.linear.y
        self.car_states[car_name][3:] = np.array([v ,phi])

        # self.ctrl_cmd_pub = rospy.Publisher('')
    def sample_sg_nodes(self):
        node_arr = [i for i in range(1, self.n_node+1)]
        # Sample nodes for start and goal
        nodes = random.sample(node_arr, 2 * self.n_car)
        start_nodes = nodes[:self.n_car]
        goal_nodes = nodes[self.n_car:]
        
        return start_nodes, goal_nodes

    def calc_yaw(self,q):
        rotation = R.from_quat(q)
        euler_angles = rotation.as_euler('zyx', degrees = True)
        yaw = euler_angles[0]
        # print("Yaw angle: ", yaw_angle, "degrees")
        return yaw






    def make_traj_msg(self, traj):
        traj = traj.copy()
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = rospy.Time.now()
        for i in range(len(traj)):
            t = rospy.Time.now()
            traj_pose = PoseStamped()
            traj_pose.header.stamp = t
            traj_pose.header.frame_id = "map"
            traj_pose.pose.position.x = traj[i][0]
            traj_pose.pose.position.y = traj[i][1]
            traj_pose.pose.position.z = 0
            # q = quaternion_from_euler(0, 0, traj[i][2])
            q = R.from_euler('zyx', [traj[i][2], 0, 0]).as_quat()
            traj_pose.pose.orientation.x = q[0]
            traj_pose.pose.orientation.y = q[1]
            traj_pose.pose.orientation.z = q[2]
            traj_pose.pose.orientation.w = q[3]            
            path.poses.append(traj_pose)
        return path

    def make_cmd_vel(self, raw_cmd):
        v,w = raw_cmd
        cmd = Twist()
        cmd.linear.x = v
        cmd.linear.y = w
        cmd.linear.z = 0
        cmd.angular.x = 0
        cmd.angular.y = 0
        cmd.angular.z = 0

        return cmd

    # def compute_control(self, H, traj, current_state):
    #     # State and control dimensions
    #     nx = 3  # State dimension [x, y, yaw]
    #     nu = 2  # Control dimension [v, w]
        
    #     orig_traj = traj
    #     traj = traj[:, :3].T

    #     current_state = current_state[:3]
    #     print("traj: ",traj)
    #     print("c state: ", current_state)
    #     # Control variables (velocity and angular velocity)
    #     v = cp.Variable((nu, H))

    #     # State variables (x, y, yaw)
    #     x = cp.Variable((nx, H + 1))

    #     # Cost function components
    #     cost = 0
    #     constraints = []

    #     for k in range(H):
    #         # Cost for state error
    #         state_error = cp.norm(x[:, k+1] - traj[:, k], 2)
    #         # Cost for control input
    #         control_input = cp.norm(v[:, k], 2)

    #         # cost += state_error**2 + control_input**2
    #         cost += state_error**2

    #         # Dynamics constraints (a simple kinematic model)
    #         # x_next = x_current + v_current * cos(yaw) * dt
    #         # y_next = y_current + v_current * sin(yaw) * dt
    #         # yaw_next = yaw_current + w_current * dt
    #         a_cos_theta = np.cos(orig_traj[k][2])
    #         a_sin_theta = np.sin(orig_traj[k][2])
    #         # x[0, k+1] == x[0, k] + v[0, k] * cp.cos(x[2, k]) * self.dt,
    #         # x[1, k+1] == x[1, k] + v[0, k] * cp.sin(x[2, k]) * self.dt,
    #         constraints += [

    #             x[0, k+1] == x[0, k] + v[0, k] * a_cos_theta * self.dt,
    #             x[1, k+1] == x[1, k] + v[0, k] * a_sin_theta * self.dt,
    #             x[2, k+1] == x[2, k] + v[1, k] * self.dt
    #         ]

    #         # Add any additional constraints, such as control input limits

    #     # Initial condition constraint
    #     constraints += [x[:,0] == current_state]

    #     # Formulate optimization problem
    #     problem = cp.Problem(cp.Minimize(cost), constraints)

    #     # Solve optimization problem
    #     problem.solve()

    #     # Extract first control input
    #     control_input = v[:,0].value
    #     return control_input        

    def compute_control(self, H, T, S):
        Sg = T[H]
        ds = np.sqrt((Sg[0]-S[0])**2+(Sg[1]-S[1])**2)
        v = ds/(H*self.dt)
        w = topi(Sg[2]-S[2])/(H*self.dt)
        return [v,w]


    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            stime = time.time()
            print(" *******************************")
            print("car state: ", self.car_states)
            H = 2
            for i in range(self.n_car):
                self.car_states[self.car_names[i]][2] = topi(self.car_states[self.car_names[i]][2])
            self.Mstates = np.array([self.car_states[car] for car in self.car_names])

            self.TM.traffic_state_update(self.Mstates)
            trajbuffer = self.TM.get_traj_buffer()
            statebuffer = self.TM.get_state_buffer()
            d_value_list = []
            for i in range(self.n_car):
                d_value = statebuffer[i]["sd"][1]
                d_value_list.append(d_value)
            cmd_list = []    
            for i in range(self.n_car):
                traj_path = self.make_traj_msg(trajbuffer[i])
                self.traj_path_pubs[self.car_names[i]].publish(traj_path)                  
                cmd = self.compute_control(min(H,len(trajbuffer[i])),trajbuffer[i].copy(), self.car_states[self.car_names[i]].copy())
                print("robot {} raw cmd: {} , d value {}".format(i+1,cmd, d_value))
                
                # cmd[1] - 10* d_value
                # if abs(cmd[1]) < 0.3 and abs(cmd[1]) > 0.03 :
                #     cmd[1] = -0.3 if cmd[1] < 0 else 0.3
                # print("d offset cmd :", cmd)
                
                cmd = [min(0.2, cmd[0]), cmd[1]*0.045]
                # cmd = [0.2, cmd[1]*0.045]
                # cmd = [0.0,-0.0]


                cmd = [0,0]


                print("robot {} final cmd : {}".format(i+1,cmd))
                cmd_list.append(cmd)
            etime = time.time()
            print("************ Time cost ********** : ", etime-stime)
            if not if_sim:
                for i in range(self.n_car):
                    twist_cmd = self.make_cmd_vel(cmd_list[i])
                    self.cmd_pubs[self.car_names[i]].publish(twist_cmd)

            rate.sleep()

if __name__ == '__main__':
    try:
        node = TrafficManagerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
