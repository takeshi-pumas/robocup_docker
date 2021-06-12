# -*- coding: utf-8 -*-

import actionlib
import cv2
import glob
import math
import moveit_commander
import numpy as np
import os
import rospy
import ros_numpy
import subprocess
import tf
import tf2_ros
import time

from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped, Twist
from IPython.display import Image
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from sensor_msgs.msg import LaserScan, PointCloud2


### Definitions and Functions ###

def screen_shot():    
    cmd = "import -window root /tmp/screen.png"
    subprocess.call(cmd.split())
    with open('/tmp/screen.png', 'rb') as file:
        display(Image(data=file.read()))

def screen_cast(sec):    
    cmd = "byzanz-record -d " + str(sec) + " /tmp/screencast.gif"
    subprocess.call(cmd.split())
    with open('/tmp/screencast.gif', 'rb') as file:
        display(Image(data=file.read()))

base_vel_pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=1)

def move_base_vel(vx, vy, vw):
    twist = Twist()
    twist.linear.x = vx
    twist.linear.y = vy
    twist.angular.z = vw / 180.0 * math.pi  
    base_vel_pub.publish(twist)  

def get_current_time_sec():
    return rospy.Time.now().to_sec()

def quaternion_from_euler(roll, pitch, yaw):    
    q = tf.transformations.quaternion_from_euler(roll / 180.0 * math.pi,
                                                 pitch / 180.0 * math.pi,
                                                 yaw / 180.0 * math.pi, 'rxyz')
    return Quaternion(q[0], q[1], q[2], q[3])

navclient = actionlib.SimpleActionClient('/move_base', MoveBaseAction)

def move_base_goal(x, y, theta):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation = quaternion_from_euler(0, 0, theta)
    navclient.send_goal(goal)
    navclient.wait_for_result()
    state = navclient.get_state()
    return True if state == 3 else False

def get_relative_coordinate(parent, child):
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    trans = TransformStamped()
    while not rospy.is_shutdown():
        try:
            trans = tfBuffer.lookup_transform(parent, child,
                                              rospy.Time().now(),
                                              rospy.Duration(4.0))
            break
        except (tf2_ros.ExtrapolationException):
            pass
    return trans.transform

whole_body = moveit_commander.MoveGroupCommander("whole_body_light")
# whole_body = moveit_commander.MoveGroupCommander("whole_body_weighted")
whole_body.allow_replanning(True)
whole_body.set_workspace([-10.0, -10.0, 10.0, 10.0])

def move_wholebody_ik(x, y, z, roll, pitch, yaw):
    p = PoseStamped()
    p.header.frame_id = "/map"
    p.pose.position.x = x
    p.pose.position.y = y
    p.pose.position.z = z
    p.pose.orientation = quaternion_from_euler(roll, pitch, yaw)
    whole_body.set_pose_target(p)
    return whole_body.go()

arm = moveit_commander.MoveGroupCommander('arm')

def move_arm_neutral():
    arm.set_named_target('neutral')
    return arm.go()

def move_arm_init():
    arm.set_named_target('go')
    return arm.go()

gripper = moveit_commander.MoveGroupCommander("gripper")

def move_hand(v):
    gripper.set_joint_value_target("hand_motor_joint", v)
    success = gripper.go()
    rospy.sleep(6)
    return success

head = moveit_commander.MoveGroupCommander("head")

def move_head_tilt(v):
    head.set_joint_value_target("head_tilt_joint", v)
    return head.go()

def get_object_dict():
    object_dict = {}
    paths = glob.glob("/opt/ros/melodic/share/tmc_wrs_gazebo_worlds/models/ycb*")
    for path in paths:
        file = os.path.basename(path)
        object_dict[file[8:]] = file
    return object_dict

def get_object_list():
    object_list = get_object_dict().values()
    object_list.sort()
    for i in range(len(object_list)):
        object_list[i] = object_list[i][8:]
    return object_list

def put_object(name, x, y, z,Y=0):
    cmd = "rosrun gazebo_ros spawn_model -database " \
          + str(get_object_dict()[name]) \
          + " -sdf -model " + str(name) \
          + " -x " + str(y - 2.1) + \
          " -y " + str(-x + 1.2) \
          + " -z " + str(z)\
          + " -Y " + str(Y)
    subprocess.call(cmd.split())

def delete_object(name):
    cmd = ['rosservice', 'call', 'gazebo/delete_model',
           '{model_name: ' + str(name) + '}']
    subprocess.call(cmd)

    
    
    
### Classes ###

class Laser():

    def __init__(self):
        self._laser_sub = rospy.Subscriber('/hsrb/base_scan', LaserScan, self._laser_cb)
        self._scan_data = None

    def _laser_cb(self, msg):        
        self._scan_data = msg

    def get_data(self):        
        return self._scan_data

    
    
    
class RGBD():

    def __init__(self):
        self._br = tf.TransformBroadcaster()
        self._cloud_sub = rospy.Subscriber(
            "/hsrb/head_rgbd_sensor/depth_registered/rectified_points",
            PointCloud2, self._cloud_cb)
        self._points_data = None
        self._image_data = None
        self._h_image = None
        self._region = None
        self._h_min = 0
        self._h_max = 0
        self._xyz = [0, 0, 0]
        self._frame_name = None

    def _cloud_cb(self, msg):
        self._points_data = ros_numpy.numpify(msg)
        self._image_data = \
            self._points_data['rgb'].view((np.uint8, 4))[..., [2, 1, 0]]
        hsv_image = cv2.cvtColor(self._image_data, cv2.COLOR_RGB2HSV_FULL)
        self._h_image = hsv_image[..., 0]
        self._region = \
            (self._h_image > self._h_min) & (self._h_image < self._h_max)
        if not np.any(self._region):
            return
        (y_idx, x_idx) = np.where(self._region)
        x = np.average(self._points_data['x'][y_idx, x_idx])
        y = np.average(self._points_data['y'][y_idx, x_idx])
        z = np.average(self._points_data['z'][y_idx, x_idx])
        self._xyz = [x, y, z]
        if self._frame_name is None:
            return
        self._br.sendTransform(
            (x, y, z), tf.transformations.quaternion_from_euler(0, 0, 0),
            rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs),
            self._frame_name,
            msg.header.frame_id)

    def get_image(self):
        return self._image_data

    def get_points(self):
        return self._points_data

    def get_h_image(self):
        return self._h_image

    def get_region(self):
        return self._region

    def get_xyz(self):
        return self._xyz

    def set_h(self, h_min, h_max):
        self._h_min = h_min
        self._h_max = h_max

    def set_coordinate_name(self, name):
        self._frame_name = name
