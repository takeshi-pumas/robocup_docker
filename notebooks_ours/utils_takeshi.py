# -*- coding: utf-8 -*-

from utils import *
import moveit_msgs.msg
import smach
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math as m
import sys

##### Publishers #####
scene_pub = rospy.Publisher('planning_scene', moveit_msgs.msg.PlanningScene, queue_size = 5)

##### KNOWN LOCATIONS #####
kl_mess1   = [1.04, 0.3, 90]
kl_tray    = [2.318411366833172, 0.09283744344925589, -90]
kl_box1    = [-0.04168256822546347, 2.427268271720426, -90]
kl_table1  = [1.04, 1.2, 90]
kl_table2  = [0 , 1.2, 90] 
kl_drawers = [0.06, 0.5, -90]


##### ARM #####
arm_grasp_from_above = [0.19263830140116414,
 -2.2668981568652917,
 -0.007358947463759424,
 -0.9939144210462025,
 -0.17365421548386273,
 0.0]
arm_grasp_from_above_table = [0.6038188787042562,
 -1.5712036013833837,
 -0.02739401124993579,
 -1.5706216522317993,
 0.00020861214307910103,
 0.0]
arm_grasp_table=[0.41349380130577407,
 -1.671584191489468,
 -0.02774372779356371,
 0.0,
 0.0,
 0.0]
arm_grasp_floor = [-1.5151551103007697e-05,
 -2.4,
 -0.2620865401925543,
 0.7019536624449207,
 0.20120924571306453,
 0.0]
arm_train_pose = [0.033749214744071214,
 -2.1204421063180217,
 -1.3982377978814715,
 -1.7296544561013807,
 2.135675364707808,
 0.0]
arm_ready_to_place = [0.03999320441056991,
 -0.4729690540086997,
 0.19361475012179108,
 -1.5269847787383313,
 -0.009753879176134461,
 0.0]
arm_high_drawer = [0.2539946870715912,
 -1.6765040634258677,
 -0.02776609034055033,
 0.0726899567834991,
 1.5763667194117463,
 0.0]

##### GRASP 
ungrasped = [-0.00047048998088961014,
 -0.03874743486886725,
 -0.04825256513113274,
 0.038463464485261056,
 -0.03874743486886725]
grasped = [0.12814103131904275,
 -0.30672794406396453,
 0.21972794406396456,
 0.13252877558892262,
 -0.30672794406396453]

def rot_to_euler(R):    
    tol = sys.float_info.epsilon * 10
    if abs(R.item(0, 0)) < tol and abs(R.item(1, 0)) < tol:
       eul1 = 0
       eul2 = m.atan2(-R.item(2, 0), R.item(0, 0))
       eul3 = m.atan2(-R.item(1, 2), R.item(1, 1))
    else:   
       eul1 = m.atan2(R.item(1, 0),R.item(0, 0))
       sp = m.sin(eul1)
       cp = m.cos(eul1)
       eul2 = m.atan2(-R.item(2, 0), cp * R.item(0, 0) + sp * R.item(1, 0))
       eul3 = m.atan2(sp * R.item(0, 2) - cp * R.item(1, 2), cp * R.item(1, 1) - sp * R.item(0, 1))
    return np.asarray((eul1, eul2, eul3))

def pca_xyz(xyz):
    quats=[]
    for i in range(len(xyz)):
        pca = PCA(n_components = 3).fit(xyz[i])
        vec0 = pca.components_[0, :]
        vec1 = pca.components_[1, :]
        vec2 = pca.components_[2, :]
        R = pca.components_
        euler = rot_to_euler(R)
        quats.append(tf.transformations.quaternion_from_euler(euler[0], euler[1], euler[2]))
    return quats

def cart2spher(x, y, z):
    ro = np.sqrt(x**2 + y**2 + z**2)
    th = np.arctan2(y, x)
    phi = np.arctan2((np.sqrt(x**2 + y**2)),z)
    return np.asarray((ro, th, phi))

def spher2cart(ro, th, phi):
    x = ro * np.cos(th) * np.sin(phi)
    y = ro * np.sin(th) * np.sin(phi)
    z = ro * np.cos(th)
    return np.asarray((x, y, z))

def point_2D_3D(points_data, px_y, px_x):
    ##px pixels /2D world  P1 3D world
    P = np.asarray((points_data[px_y, px_x]['x'], points_data[px_y, px_x]['y'], points_data[px_y, px_x]['z']))
    return P

#Setting points
def set_points(points_data, px_y, px_x):
    P = np.asarray((points_data[px_y, px_x]['x'], points_data[px_y, px_x]['y'], points_data[px_y, px_x]['z']))
    return P
