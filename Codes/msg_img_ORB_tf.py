# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 15:17:33 2020

@author: oscar
"""

#!/usr/bin/env python


import cv2
from geometry_msgs.msg import Quaternion
import numpy as np
import rospy
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseStamped
from tf.transformations import *
from tf.msg import tfMessage
from tf.transformations import euler_from_quaternion
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()
global cap_cnt


from std_msgs.msg import String
def shutdown(self):
 
        exit()

def callback(img_msg):
    cap_cnt=0
   
   
    
################################################################ DO THINGS HERE
    cv2_img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    gray = cv2.cvtColor(cv2_img,cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp = orb.detect(cv2_img,None)
    
    kp, des = orb.compute(cv2_img, kp)
    img_ref= cv2.imread("capture_1.png", cv2.IMREAD_COLOR)
    kp_r, des_r = orb.compute(img_ref, kp)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches=matcher.match(des,des_r)
    matches.sort(key=lambda x: x.distance, reverse=False)
    numGoodMatches = int(len(matches) * .50)
    matches = matches[:numGoodMatches]
    print (len(matches))
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = kp[match.queryIdx].pt
        points2[i, :] = kp_r[match.trainIdx].pt  
  # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    print(h)
    imMatches= cv2.drawMatches(cv2_img,kp,img_ref,kp_r, matches, None)
       #-- Get the corners from the image_1 ( the object to be "detected" )
    obj_corners = np.empty((4,1,2), dtype=np.float32)
    obj_corners[0,0,0] = 0
    obj_corners[0,0,1] = 0
    obj_corners[1,0,0] = cv2_img.shape[1]
    obj_corners[1,0,1] = 0
    obj_corners[2,0,0] = cv2_img.shape[1]
    obj_corners[2,0,1] = cv2_img.shape[0]
    obj_corners[3,0,0] = 0
    obj_corners[3,0,1] = cv2_img.shape[0]
    scene_corners = cv2.perspectiveTransform(obj_corners, h)
    cv2.line(imMatches, (int(scene_corners[0,0,0] + cv2_img.shape[1]), int(scene_corners[0,0,1])),\
    (int(scene_corners[1,0,0] + cv2_img.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)
    cv2.line(imMatches, (int(scene_corners[1,0,0] + cv2_img.shape[1]), int(scene_corners[1,0,1])),\
    (int(scene_corners[2,0,0] + cv2_img.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
    cv2.line(imMatches, (int(scene_corners[2,0,0] + cv2_img.shape[1]), int(scene_corners[2,0,1])),\
    (int(scene_corners[3,0,0] + cv2_img.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
    cv2.line(imMatches, (int(scene_corners[3,0,0] + cv2_img.shape[1]), int(scene_corners[3,0,1])),\
    (int(scene_corners[0,0,0] + cv2_img.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)
    
    
    
     
    
    img2 = cv2.drawKeypoints(cv2_img, kp, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    

   
    
    
   # np.save('imagen.npy',cv2_img)

    """circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    if circles is not None:
        print        
        print (circles)"""
#################  
    cv2.namedWindow("hand_cam")
    cv2.imshow("hand_cam", imMatches)    
    k = cv2.waitKey(1)
    if k & 0xFF == ord('q'):
        # q key pressed so quit
        print("Quitting...")
        kill_node=True
        cv2.destroyAllWindows()
        exit()

    elif k & 0xFF == ord('c'):
        # c key pressed so capture frame to image file
                
        cap_cnt=cap_cnt+1
        cap_name = "capture_{}.png".format(cap_cnt)
        cv2.imwrite(cap_name, cv2_img)
        print("Saving {}!".format(cap_name))
        

    
    
    
def listener():
    

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('hand_camera_listener', anonymous=True)
    rospy.on_shutdown(shutdown)

    rospy.Subscriber("/hsrb/hand_camera/image_raw", Image, callback)
    global kill_node
    kill_node=False
    # spin() simply keeps python from exiting until this node is stopped
    if not kill_node:
        rospy.spin()
    


if __name__ == '__main__':
    listener()
