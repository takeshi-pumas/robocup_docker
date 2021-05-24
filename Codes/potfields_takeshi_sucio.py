#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 22:54:18 2019

@author: oscar
"""

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist , PointStamped
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan
import numpy as np
from utils import *
xcl,ycl=0,0
tfBuffer = tf2_ros.Buffer()




def normalize(x,y):
    xn= x/np.linalg.norm((x,y))
    yn= y/np.linalg.norm((x,y))
    return ((xn,yn))

def newOdom (msg):
    global x
    global y
    global th
    
    x=msg.pose.pose.position.x
    y=msg.pose.pose.position.y
    quaternion = (
    msg.pose.pose.orientation.x,
    msg.pose.pose.orientation.y,
    msg.pose.pose.orientation.z,
    msg.pose.pose.orientation.w)
    euler = euler_from_quaternion(quaternion)
    th=euler[2]


def readPoint(punto):
    global xcl
    global ycl
    xcl =punto.point.x
    ycl =punto.point.y
    print ( 'I READ',xcl,ycl)
    print ('Lets try' ,gazebo_2_world(xcl,ycl))
    
def readSensor(data):
    lec=np.asarray(data.ranges)
    lec[np.isinf(lec)]=13.5
    
    Fx, Fy,Fth = 0.001,0.001,0
     


     #deltaang=4.7124/len(data.ranges)
    deltaang=4.18/len(data.ranges) 
    laserdegs=  np.arange(-2.09,2.09,deltaang)
    print ('degree',laserdegs[0],lec[0], lec[360],lec[-1])
    Fx=0
    Fy = 0.001
    for i,deg in enumerate(laserdegs):
        
        if (lec[i] < 2):
            Fx = Fx + (1/lec[i])**2 * np.cos(deg)
            Fy = Fy + (1/lec[i])**2 * np.sin(deg)
     
 	Fth= np.arctan2(Fy,(Fx+.000000000001))+np.pi
 	Fmag= np.linalg.norm((Fx,Fy))
 	if Fth > np.pi :
 		Fth+=(-2*np.pi)
    
	
    print('Frep x , y ,Fmag degrees',Fx,Fy,Fmag,Fth*180/np.pi)
    
    xy,xycl=np.array((x,y)) ,   np.array((xcl,ycl))
    euclD=np.linalg.norm(xy-xycl)
    #trans = tfBuffer.lookup_transform( 'base_footprint','map', rospy.Time())

    print("xrob,yrob, throbot",x,y,th*180/3.1416)
    print(whole_body.get_current_pose())
    print("xclick,yclick",xcl,ycl,"euclD",euclD)
    Fatrx =( -x + xcl)/euclD**2
    Fatry =( -y + ycl)/euclD**2      
    Fatrth=np.arctan2(Fatrx, Fatry) 
    Fatrth=Fatrth#-th
    Fmagat= np.linalg.norm((Fatrx,Fatry))

    print ('Fatx, Fatry,Fmagat, Fatrth',Fatrx,Fatry,Fmagat,(Fatrth)*180/np.pi )

    Ftotx=   Fmag*np.cos(Fth) *.1  +    Fmagat*np.cos(Fatrth) #Fx
    Ftoty=   Fmag*np.sin(Fth) *.1  +    Fmagat*np.sin(Fatrth) #Fy
    
    Ftotth=  np.arctan2(Ftoty,Ftotx)	 
    if Ftotth > np.pi :
    
    	Ftotth=-np.pi-(np.pi - Ftotth)
    


    print('Ftotxy',Ftotx,Ftoty,Ftotth*180/np.pi)
    Fatmag=np.linalg.norm((Fatrx,Fatry))
    Fmag=np.linalg.norm((Fx,Fy))
    #print ("theta robot",th*180/3.1416,'---------------------------')
    #print ('fasorFatrth',np.linalg.norm((Fatrx,Fatry)),(Fatrth)*180/3.1416 )
    #print ("FXATR,FYATR",Fatrx,Fatry)
    #print ('fasorFrepth',np.linalg.norm((Fx,Fy)),Fth*180/3.1416)
    #print ("Frepx,Frepy",Fx,Fy)
    """Fx,Fy= Fmag*np.cos(Fth) , Fmag*np.sin(Fth)   
    Fatrx,Fatry= Fatmag*np.cos(Fatrth) , Fatmag*np.sin(Fatrth) """
    vel=.01
    if (Fmag >90):
        if( abs(Ftotth) >= (0.75*np.pi)):

            #speed.linear.x=-0
            print('lin over pi /2')
            #speed.angular.z=0.15


        if( abs(Ftotth) <= (0.25*np.pi)):
                        speed.linear.x=1.0
                        print('lin')
                        speed.angular.z=0
        if ( abs(Ftotth) > (0.25*np.pi) and    abs(Ftotth) < (0.75*np.pi)             )  :
           
             
            if Ftotth < 0:
                if (abs( Ftotth ) < np.pi/2):
                    vel=.24
                    print('open curve')
             
                print('Vang-')
                speed.linear.x=vel
                speed.angular.z=-0.55
            if Ftotth > 0:
                if (abs( Ftotth ) < np.pi/2):
                    vel=.24
                    print('open curve')
             
                
                print('Vang+')
                speed.linear.x=vel
                speed.angular.z=.5
    else:
        print('Fields too weak')
        speed.linear.x=1.9
        print('lin ++ ')
        speed.angular.z=0
    speed.linear.x=0
    speed.angular.z=0   
speed=Twist()
speed.angular.z=0
def inoutinout():
    sub= rospy.Subscriber("/hsrb/odom_ground_truth",Odometry,newOdom)
    sub2=rospy.Subscriber("/hsrb/base_scan",LaserScan,readSensor)
    sub3=rospy.Subscriber("/clicked_point",PointStamped,readPoint)
    pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=1)
    rospy.init_node('talker_cmdvel', anonymous=True)
    rate = rospy.Rate(15) # 10hz
    while not rospy.is_shutdown():
        hello_str = "Time %s" % rospy.get_time()
        #print("odom",x,y,th)
  #    
        pub.publish(speed)
        rate.sleep()

if __name__ == '__main__':
    try:
        inoutinout()
    except rospy.ROSInterruptException:
        pass
