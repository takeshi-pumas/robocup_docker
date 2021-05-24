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


from std_msgs.msg import String
def shutdown(self):
 
        exit()

def callback(img_msg):
        print( "got image")
       
        
    ################################################################ DO THINGS HERE
        cv2_img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
        gray = cv2.cvtColor(cv2_img,cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        kp = orb.detect(cv2_img,None)
        
        kp, des = orb.compute(cv2_img, kp)
        img2 = cv2.drawKeypoints(cv2_img, kp, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
       
        
        
       # np.save('imagen.npy',cv2_img)
    
        """circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
        if circles is not None:
            print        
            print (circles)"""
    #################  
        cv2.namedWindow("xtion_cam")
        #cv2.imshow("hand_cam", img2)
        cv2.imshow("xtion_cam", cv2_img)    
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            # q key pressed so quit
            print("Quitting...")
            kill_node=True
            cv2.destroyAllWindows()
            exit()
    
        elif k & 0xFF == ord('c'):
            # c key pressed so capture frame to image file
            cap_cnt=0
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

    #rospy.Subscriber("/hsrb/hand_camera/image_raw", Image, callback)
    rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_rect_color", Image, callback)
    
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    


if __name__ == '__main__':
    listener()
