# -*- coding: utf-8 -*-

from utils import *
import moveit_msgs.msg
#from gazebo_ros import gazebo_interface
import smach
import matplotlib.pyplot as plt


##### Publishers #####
scene_pub = rospy.Publisher('planning_scene', moveit_msgs.msg.PlanningScene, queue_size = 5)

##### KNOWN LOCATIONS #####
kl_mess1 = [1.04, 0.3, 90]
kl_tray =  [2.318411366833172, 0.09283744344925589, -90]
kl_box1 =  [-0.04168256822546347, 2.427268271720426, -90]
kl_table1 = [1.04, 1.2, 90]
kl_table2= [0 , 1.2,90] 
kl_drawers=  [0.06, 0.5, -90]


##### ARM #####
arm_grasp_from_above = [0.19263830140116414,
 -2.2668981568652917,
 -0.007358947463759424,
 -0.9939144210462025,
 -0.17365421548386273,
 0.0]
arm_grasp_from_above_table = [0.41349380130577407,
 -1.671584191489468,
 -0.02774372779356371,
 -1.5952436225825641,
 0.22362492457833927,
 0.0]
arm_grasp_table=[0.41349380130577407,
 -1.671584191489468,
 -0.02774372779356371,
 0.0,
 0.22362492457833927,
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
arm_high_drawer=[0.2539946870715912,
 -1.6765040634258677,
 -0.02776609034055033,
 0.0726899567834991,
 1.5763667194117463,
 0.0]

##### GRASP 
ungrasped=[-0.00047048998088961014,
 -0.03874743486886725,
 -0.04825256513113274,
 0.038463464485261056,
 -0.03874743486886725]
grasped=[0.12814103131904275,
 -0.30672794406396453,
 0.21972794406396456,
 0.13252877558892262,
 -0.30672794406396453]



########## Base Class for takeshi state machine ##########
class Takeshi_states(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ['succ', 'failed', 'tries'])
        self.tries = 1
        self.max_tries = 5
    
    #This function must be overwritten
    def takeshi_run(self):
        print("Takeshi execute")
        success = True        
        return success

    def execute(self, userdata):
        #rospy.loginfo('STATE : INITIAL')
        print("Excecuting Takeshi State: " + self.__class__.__name__)
        print('Try', self.tries, 'of 5 attempts')
        succ = self.takeshi_run()
        if succ: 
            print('success')            
            return 'succ'
        else:
            print('failed')
            self.tries += 1
            if self.tries > self.max_tries:
                return 'tries'
            else:
                return 'failed'


########## Util Functions ##########

def cart2spher(x,y,z):
    ro= np.sqrt(x**2+y**2+z**2)
    th=np.arctan2(y,x)
    phi=np.arctan2((np.sqrt(x**2+y**2)),z)
    return np.asarray((ro,th,phi))


def spher2cart(ro,th,phi):
    x= ro * np.cos(th)* np.sin(phi)
    y= ro * np.sin(th)* np.sin(phi)
    z= ro*  np.cos(th)
    return np.asarray((x,y,z))

def point_2D_3D(points_data, px_y, px_x):
    ##px pixels /2D world  P1 3D world
    P = np.asarray((points_data[px_y, px_x]['x'], points_data[px_y, px_x]['y'], points_data[px_y, px_x]['z']))
    return P
def segment_floor():
    image_data = rgbd.get_image()
    points_data = rgbd.get_points()

    P1 = point_2D_3D(points_data, -1, -200)
    P2 = point_2D_3D(points_data, -1, 200)
    P3 = point_2D_3D(points_data, -150, -320)

    V1 = P1 - P2
    V2 = P3 - P2
    nx, ny, nz = np.cross(V2, V1)
    print('look at the phi angle  in normal vector', np.rad2deg(cart2spher(nx, ny, nz))[2] - 90)
    trans, rot = listener.lookupTransform('/map', '/head_rgbd_sensor_gazebo_frame', rospy.Time(0))
    euler = tf.transformations.euler_from_quaternion(rot)
    print(np.rad2deg(euler)[1], ' if this degree is not the same as head tilt plane was not found')
       
    mask = np.zeros((image_data.shape))
    plane_mask = np.zeros((image_data.shape[0], image_data.shape[1]))
    mask[:, :, 0] = points_data['x'] - P1[0]
    mask[:, :, 1] = points_data['y'] - P1[1]
    mask[:, :, 2] = points_data['z'] - P1[2]
    
    for i in range (image_data.shape[0]):
        for j in range (image_data.shape[1]):
            plane_mask[i, j] = -np.dot(np.asarray((nx, ny, nz, )), mask[i, j])
    plane_mask = plane_mask - np.min(plane_mask)
    plane_mask = plane_mask * 256 / np.max(plane_mask)
    plane_mask.astype('uint8')

    ret, thresh = cv2.threshold(plane_mask, 3, 255, 0)

    cv2_img = plane_mask.astype('uint8')
    img = plane_mask.astype('uint8')
    _, contours, hierarchy = cv2.findContours(thresh.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    cents = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        if area > 200 and area < 50000 :                        
            boundRect = cv2.boundingRect(contour)            
            img = cv2.rectangle(img, (boundRect[0], boundRect[1]), (boundRect[0] + boundRect[2], boundRect[1] + boundRect[3]), (255, 0, 0), 2)
            # calculate moments for each contour
            xyz = []
                
            for jy in range (boundRect[0], boundRect[0] + boundRect[2]):
                for ix in range(boundRect[1], boundRect[1] + boundRect[3]):
                    xyz.append(np.asarray((points_data['x'][ix, jy], points_data['y'][ix, jy], points_data['z'][ix, jy])))
            xyz = np.asarray(xyz)
            cent = xyz.mean(axis = 0)
            
            cents.append(cent)

            M = cv2.moments(contour)
            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
            cv2.putText(img, "centroid_" + str(i) + "_" + str(cX) + ',' + str(cY), (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cents = np.asarray(cents)    
    return cents  


def static_tf_publish(cents):
    for  i, cent  in enumerate(cents):
        x, y, z = cent
        print(cent,i)
        broadcaster.sendTransform((x, y, z), rot, rospy.Time.now(), 'Closest_Object' + str(i), "head_rgbd_sensor_link")
        rospy.sleep(0.2)
        xyz_map, cent_quat = listener.lookupTransform('/map', 'Closest_Object' + str(i), rospy.Time(0))
        print(xyz_map[0],i)
        map_euler = tf.transformations.euler_from_quaternion(cent_quat)
        rospy.sleep(0.2)
        static_transformStamped = TransformStamped()

        ##FIXING TF TO MAP ( ODOM REALLY)    
        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = "map"
        static_transformStamped.child_frame_id = "static" + str(i)
        static_transformStamped.transform.translation.x = float(xyz_map[0])
        static_transformStamped.transform.translation.y = float(xyz_map[1])
        static_transformStamped.transform.translation.z = float(xyz_map[2])
        static_transformStamped.transform.rotation.x = 0    
        static_transformStamped.transform.rotation.y = 0    
        static_transformStamped.transform.rotation.z = 0    
        static_transformStamped.transform.rotation.w = 1    

        tf_static_broadcaster.sendTransform(static_transformStamped)
    return True


def add_object(name, size, pose, orientation):
    p = PoseStamped()
    p.header.frame_id = "map"       # "head_rgbd_sensor_link"
    
    p.pose.position.x = pose[0]
    p.pose.position.y = pose[1]
    p.pose.position.z = pose[2]

    p.pose.orientation.x = orientation[0] * np.pi
    p.pose.orientation.w = orientation[1] * np.pi

    scene.add_box(name, p, size)


def publish_scene():
    add_object("shelf", [1.5, 0.04, 0.4], [2.5, 4.7, 0.78], [0.5, 0.5])
    add_object("shelf1", [1.5, 0.04, 0.4], [2.5, 4.7, 0.49], [0.5, 0.5])
    add_object("shelf2", [1.5, 0.04, 0.4], [2.5, 4.7, 0.18], [0.5, 0.5])
    add_object("shelf_wall", [1, 1, 0.04], [2.5, 4.9, 0.5], [0.5, 0.5])
    add_object("shelf_wall1", [.04, 1, 0.4], [2.7, 4.9, 0.5], [0.5, 0.5])
    add_object("shelf_wall2", [.04, 1, 0.4], [1.8, 4.9, 0.5], [0.5, 0.5])
    
    add_object("table_big", [1.7, 0.13, 0.7], [0.95, 1.9, 0.34], [0.5, 0.5])
    add_object("table_small", [0.5, 0.01, 0.4], [0.1, 1.9, 0.61], [0.5, 0.5])
    add_object("table_tray", [0.65, 0.01, 0.7], [1.8, -0.65, 0.4], [0.5, 0.5])  
    
    static_transformStamped = TransformStamped()

    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = "map"
    static_transformStamped.child_frame_id = "Drawer_high" 
    static_transformStamped.transform.translation.x = 0.14
    static_transformStamped.transform.translation.y = -0.344
    static_transformStamped.transform.translation.z = 0.57
    static_transformStamped.transform.rotation.x = 0    
    static_transformStamped.transform.rotation.y = 0    
    static_transformStamped.transform.rotation.z = 0    
    static_transformStamped.transform.rotation.w = 1    

    tf_static_broadcaster.sendTransform(static_transformStamped)


      ##FIXING TF TO MAP ( ODOM REALLY)    
    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = "map"
    static_transformStamped.child_frame_id = "Drawer_low" 
    static_transformStamped.transform.translation.x = 0.14
    static_transformStamped.transform.translation.y = -0.344
    static_transformStamped.transform.translation.z = 0.27
    static_transformStamped.transform.rotation.x = 0    
    static_transformStamped.transform.rotation.y = 0    
    static_transformStamped.transform.rotation.z = 0    
    static_transformStamped.transform.rotation.w = 1    

    tf_static_broadcaster.sendTransform(static_transformStamped)
    ##FIXING TF TO MAP ( ODOM REALLY)    
    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = "map"
    static_transformStamped.child_frame_id = "Box1" 
    static_transformStamped.transform.translation.x = 2.4
    static_transformStamped.transform.translation.y = -0.6
    static_transformStamped.transform.translation.z = .5
    static_transformStamped.transform.rotation.x = 0    
    static_transformStamped.transform.rotation.y = 0    
    static_transformStamped.transform.rotation.z = 0    
    static_transformStamped.transform.rotation.w = 1    

    tf_static_broadcaster.sendTransform(static_transformStamped)  
    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = "map"
    static_transformStamped.child_frame_id = "Drawer_left" 
    static_transformStamped.transform.translation.x = .45
    static_transformStamped.transform.translation.y = -0.33
    static_transformStamped.transform.translation.z = .28
    static_transformStamped.transform.rotation.x = 0    
    static_transformStamped.transform.rotation.y = 0    
    static_transformStamped.transform.rotation.z = 0    
    static_transformStamped.transform.rotation.w = 1    

    return True

def segment_table2(chan):
    image_data=rgbd.get_image()
    points_data = rgbd.get_points()

    mask=np.zeros((image_data.shape))
    plane_mask=np.zeros((image_data.shape[0],image_data.shape[1]))

    plane_mask=image_data[:,:,chan]

    ret,thresh = cv2.threshold(image_data[:,:,2],240,255,200)
    plane_mask=points_data['z']
    cv2_img=plane_mask.astype('uint8')
    img=image_data[:,:,0]
    _,contours, hierarchy = cv2.findContours(thresh.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    i=0
    cents=[]
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area > 200 and area < 50000 :
            print('contour',i,'area',area)

            boundRect = cv2.boundingRect(contour)
            #just for drawing rect, dont waste too much time on this
            print boundRect
            img=cv2.rectangle(img,(boundRect[0], boundRect[1]),(boundRect[0]+boundRect[2], boundRect[1]+boundRect[3]), (0,0,0), 2)
            # calculate moments for each contour
            xyz=[]


            for jy in range (boundRect[0], boundRect[0]+boundRect[2]):
                for ix in range(boundRect[1], boundRect[1]+boundRect[3]):
                    xyz.append(np.asarray((points_data['x'][ix,jy],points_data['y'][ix,jy],points_data['z'][ix,jy])))
            xyz=np.asarray(xyz)
            cent=xyz.mean(axis=0)
            cents.append(cent)
            M = cv2.moments(contour)
            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
            cv2.putText(img, "centroid_"+str(i)+"_"+str(cX)+','+str(cY)    ,    (cX - 25, cY - 25)   ,cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            print ('cX,cY',cX,cY)
    cents=np.asarray(cents)
    
    return (cents)
def segment_table():
    image_data=rgbd.get_image()
    points_data = rgbd.get_points()

    mask=np.zeros((image_data.shape))
    plane_mask=np.zeros((image_data.shape[0],image_data.shape[1]))

    plane_mask=image_data[:,:,1]

    ret,thresh = cv2.threshold(image_data[:,:,2],240,255,200)
    plane_mask=points_data['z']
    cv2_img=plane_mask.astype('uint8')
    img=image_data[:,:,0]
    _,contours, hierarchy = cv2.findContours(thresh.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    i=0
    cents=[]
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area > 2000 and area < 50000 :
            print('contour',i,'area',area)

            boundRect = cv2.boundingRect(contour)
            #just for drawing rect, dont waste too much time on this
            print boundRect
            img=cv2.rectangle(img,(boundRect[0], boundRect[1]),(boundRect[0]+boundRect[2], boundRect[1]+boundRect[3]), (0,0,0), 2)
            # calculate moments for each contour
            xyz=[]


            for jy in range (boundRect[0], boundRect[0]+boundRect[2]):
                for ix in range(boundRect[1], boundRect[1]+boundRect[3]):
                    xyz.append(np.asarray((points_data['x'][ix,jy],points_data['y'][ix,jy],points_data['z'][ix,jy])))
            xyz=np.asarray(xyz)
            cent=xyz.mean(axis=0)
            cents.append(cent)
            M = cv2.moments(contour)
            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
            cv2.putText(img, "centroid_"+str(i)+"_"+str(cX)+','+str(cY)    ,    (cX - 25, cY - 25)   ,cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            print ('cX,cY',cX,cY)
    cents=np.asarray(cents)
    return (cents)
