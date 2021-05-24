#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 22:54:18 2019

@author: oscar
"""

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist , PointStamped , Pose
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan
import numpy as np
import tf2_ros



from utils import *
xcl,ycl=0,0

x,y,th = 0,0,0




def normalize(x,y):
    xn= x/np.linalg.norm((x,y))
    yn= y/np.linalg.norm((x,y))
    return ((xn,yn))




def readPoint(punto):
    global xcl
    global ycl
    xcl =punto.point.x
    ycl =punto.point.y
    print ( 'I READ',xcl,ycl)
  
    
def readSensor(data):
    global x 
    global y
    global th
    lec=np.asarray(data.ranges)
    lec[np.isinf(lec)]=13.5
    
    Fx, Fy,Fth = 0.001,0.001,0
     

    #deltaang=4.7124/len(data.ranges)
    deltaang=4.18/len(data.ranges) 
    laserdegs=  np.arange(-2.09,2.09,deltaang)
    Fx=0
    Fy = 0.001
    for i,deg in enumerate(laserdegs):
        
        if (lec[i] < 0.75):
            Fx = Fx + (1/lec[i])**2 * np.cos(deg)
            Fy = Fy + (1/lec[i])**2 * np.sin(deg)
     
 	Fth= np.arctan2(Fy,(Fx+.000000000001))+np.pi
 	Fmag= np.linalg.norm((Fx,Fy))
    
 	#if Fth > np.pi :
 	#	Fth+=(-2*np.pi)
    
	
   
    xy,xycl=np.array((x,y)) ,   np.array((xcl,ycl))
    euclD=np.linalg.norm(xy-xycl)
    pose_robot=Pose()

    
      
    #print("xrob,yrob, throbot",x,y,th*180/3.1416)
    x,y = trans.transform.translation.x , trans.transform.translation.y#trans.transform.translation.x , trans.translation.y # pose_robot.pose.position.x , pose_robot.pose.position.y
    euler= tf.transformations.euler_from_quaternion((trans.transform.rotation.x , trans.transform.rotation.y, trans.transform.rotation.z , trans.transform.rotation.w)) 
    #print (pose_robot)
    #print (euler)
    th= euler[2]
    #uler tf.transformations.euler_from_quaternion( (pose_robot.pose.orientation)             )
    
    print("x rob map,y rob map , throbot",x,y,th*180/3.1416)

    print("xclick,yclick",xcl,ycl,"euclD",euclD)
    Fatrx =( -x + xcl)/euclD**2
    Fatry =( -y + ycl)/euclD**2      
    Fatrth=np.arctan2(Fatry, Fatrx) 
    Fatrth=Fatrth - th
    Fmagat= np.linalg.norm((Fatry,Fatrx))
   
    katr=100
    krep=.1
    print ('Fmagrep, Freprth',krep* Fmag,(Fth)*180/np.pi )
    print ('Fmagat, Fatrth', katr* Fmagat,(Fatrth)*180/np.pi )
    Ftotx=   Fmag*np.cos(Fth) *krep  +     katr * Fmagat*np.cos(Fatrth) #Fx
    Ftoty=   Fmag*np.sin(Fth) *krep  +     katr * Fmagat*np.sin(Fatrth) #Fy
    
    Ftotth=  np.arctan2(Ftoty,Ftotx)	 
    if ( Ftotth> np.pi ):
        Ftotth=       -np.pi-    (Ftotth-np.pi)
    if (Ftotth < -np.pi):
        Ftotth= (Ftotth     +2 *np.pi)

    print('Ftot Mag ang',np.linalg.norm((Ftotx,Ftoty)),Ftotth*180/np.pi)
    Fatmag=np.linalg.norm((Fatrx,Fatry))
    Fmag=np.linalg.norm((Fx,Fy))
    #print ("theta robot",th*180/3.1416,'---------------------------')
    #print ('fasorFatrth',np.linalg.norm((Fatrx,Fatry)),(Fatrth)*180/3.1416 )
    #print ("FXATR,FYATR",Fatrx,Fatry)
    #print ('fasorFrepth',np.linalg.norm((Fx,Fy)),Fth*180/3.1416)
    #print ("Frepx,Frepy",Fx,Fy)
    """Fx,Fy= Fmag*np.cos(Fth) , Fmag*np.sin(Fth)   
    Fatrx,Fatry= Fatmag*np.cos(Fatrth) , Fatmag*np.sin(Fatrth) """

    
    if( abs(Ftotth) <= (0.25*np.pi)):
        speed.linear.x=2.0
        print('lin')
        speed.angular.z=0
    else:
        if( Ftotth > (0.25*np.pi)):
            speed.linear.x=0.0
            print('Vang + ')
            speed.angular.z=0.5
        if( Ftotth <(0.25*np.pi)):
            speed.linear.x=0.0
            print('Vang - ')
            speed.angular.z=-0.5
    
    

    ##if ( abs(Ftotth) > (0.25*np.pi) and    abs(Ftotth) < (0.95*np.pi)             )  :
    #    if Ftotth < 0:
    #        if (abs( Ftotth ) < np.pi/2):
    #            vel=0
    #            print('open curve')
    #     
    #        print('Vang-')
    #        speed.linear.x=vel
    #        speed.angular.z=-0.55
    #    if Ftotth > 0:
    #        if (abs( Ftotth ) < np.pi/2):
    #            vel=0
    #            print('open curve')
    #     
    #        
    #        print('Vang+')
    #        speed.linear.x=vel
    #        speed.angular.z=.5
    

speed=Twist()
def inoutinout():

 
    global trans
    #sub= rospy.Subscriber("/hsrb/odom_ground_truth",Odometry,newOdom)
    rospy.init_node('talker_cmdvel', anonymous=True)
    sub2=rospy.Subscriber("/hsrb/base_scan",LaserScan,readSensor)
    sub3=rospy.Subscriber("/clicked_point",PointStamped,readPoint)
    pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=1)
    rate = rospy.Rate(15) # 10hz
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    
    
    while not rospy.is_shutdown():
        hello_str = "Time %s" % rospy.get_time()
        try:
            trans = tfBuffer.lookup_transform( 'map', 'base_footprint', rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue

        pub.publish(speed)
        rate.sleep()








if __name__ == '__main__':
    try:
        inoutinout()
    except rospy.ROSInterruptException:
        pass
