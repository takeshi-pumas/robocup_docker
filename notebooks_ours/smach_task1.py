#!/usr/bin/env python

from utils_takeshi import *

########## Functions for takeshi states ##########
class Proto_state(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        rospy.loginfo('State : PROTO_STATE')

        if self.tries==3:
            self.tries=0 
            return'tries'
        if succ:
            return 'succ'
        else:
            return 'failed'
        global trans_hand
        move_hand(1)
        self.tries+=1
        if self.tries==3:
            self.tries=0 
            return'tries'

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
    
    static_transformStamped=TransformStamped()

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

########## Clases derived from Takeshi_states, please only define takeshi_run() ##########

##### Define state INITIAL #####
#Estado inicial de takeshi, neutral
class Initial(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0

        
    def execute(self,userdata):

        
        rospy.loginfo('STATE : Going to known location Mess 1')
        print('Try',self.tries,'of 5 attepmpts') 
        self.tries+=1
        scene.remove_world_object()
        #Takeshi neutral
        move_hand(0)
        arm.set_named_target('go')
        arm.go()
        head.set_named_target('neutral')
        succ = head.go()             
        if succ:
            return 'succ'
        else:
            return 'failed'

##### Define state SCAN_FLOOR #####
#Va al mess1 piso y voltea hacia abajo la cabeza y escanea el piso
class scan_mess(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    
    def execute(self,userdata):
        
        rospy.loginfo('State : SCAN_MESS')
        print('Try',self.tries,'of 5 attepmpts')       
        head_val=head.get_current_joint_values()
        head_val[0]=np.deg2rad(0)
        head_val[1]=np.deg2rad(-45)
        #WATCH OUT FOR JOINTS LIMITS (exorcist joke)
        #plan and execute target pose
        head.set_joint_value_target(head_val)
        succ=head.go()

        if succ: 
            #trans , rot = listener.lookupTransform('/map', '/head_rgbd_sensor_gazebo_frame', rospy.Time(0))
#
            #euler=tf.transformations.euler_from_quaternion(rot)
            #print('xtion_tf',trans, euler)


            print('succ')
            return 'succ'
        else:
            print('failed')
            self.tries+=1
            if tries == 6:
                return self.tries
            else:
                return 'failed'




class Scan_floor(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries','change'])
        self.tries=0
    def execute(self,userdata):
        self.tries+=1
        print(self.tries,'of 3')
        global cents, rot, trans
        goal_x , goal_y, goal_yaw = kl_mess1        
        succ = move_base_goal(goal_x, goal_y, goal_yaw)        
        head_val = head.get_current_joint_values()
        head_val[0] = np.deg2rad(0)
        head_val[1] = np.deg2rad(-45)        
        head.go(head_val)
        trans, rot = listener.lookupTransform('/map', '/head_rgbd_sensor_gazebo_frame', rospy.Time(0))
        euler = tf.transformations.euler_from_quaternion(rot)
        #print(trans, euler)
        cents = segment_floor()
        if self.tries==3:
            self.tries=0 
            print('lets try table now')
            return'tries'

        else:
            if len (cents)==0:
                return 'failed'
            static_tf_publish(cents)
            self.tries=0 

            return 'succ'

##### Define state PRE_FLOOR #####
#Baja el brazo al suelo, abre la garra y se acerca al objeto para grasp
class Pre_floor(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['counter_in'],output_keys=['counter_out'])
        self.tries=0
    def execute(self,userdata):
        self.tries+=1
        if self.tries==3:
            self.tries=0 
            print('lets try table now')
            move_hand(0)
           
            arm.set_named_target('go')
            arm.go()
            head.set_named_target('neutral')
            succ = head.go()        
            return'tries'

        global closest_cent
        move_hand(1)
        publish_scene()
              
        trans_cents = []
        
        for i, cent in enumerate(cents):
            trans_map, _ = listener.lookupTransform('/map', 'static' + str(i), rospy.Time(0))
            trans_cents.append(trans_map)
        
        np.linalg.norm(np.asarray(trans_cents) - trans , axis = 1)
        closest_cent = np.argmin(np.linalg.norm(np.asarray(trans_cents) - trans , axis = 1))
        xyz=np.asarray(trans_cents[closest_cent])
        print (xyz)
        
        print('risk it in 5',userdata.counter_in)
        
        if(userdata.counter_in > 5):
            print ("YOOOLOOOOOOOOOOO (not the algorithm")
        else:


            if  (xyz[0] < 0.35) and (xyz[0]  >1.8):
                print ('Path to table clear,,, try first ')
                arm.set_named_target('go')
                arm.go()
                head.set_named_target('neutral')
                head.go()             
                self.tries ==5
                return 'tries'
            
            if  (xyz[1] > 1.55) and userdata.counter_in < 2  :  #<
                print ('Too risky try table first ')
                arm.set_named_target('go')
                arm.go()
                head.set_named_target('neutral')
                head.go()             
                self.tries ==5
                return 'tries'



        arm.go(arm_grasp_floor)  
        trans_hand, rot_hand = listener.lookupTransform('/hand_palm_link', 'static' + str(closest_cent), rospy.Time(0))
        wb = whole_body.get_current_joint_values()
        wb[0] += trans_hand[2] -0.1
        wb[1] += trans_hand[1]-.05
        try:
            succ = whole_body.go(wb)
        except MoveItCommanderException:
            return  'failed'
        trans_hand, rot_hand = listener.lookupTransform('/hand_palm_link', 'static' + str(closest_cent), rospy.Time(0))
        wb = whole_body.get_current_joint_values()
        wb[0] += trans_hand[2] -0.05
        wb[1] += trans_hand[1]
        try:
            succ = whole_body.go(wb)
        except MoveItCommanderException:
            return  'failed'
        succ = whole_body.go(wb)

        if succ:
            return 'succ'
        else:
            return 'failed'


##### Define state GRASP_FLOOR #####
#Se acerca mas al objeto y cierra la garra
class Grasp_floor(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        global trans_hand
        move_hand(1)
        #self.tries+=1
        #if self.tries==3:
        #    arm.set_named_target('go')
        #    arm.go()
        #    head.set_named_target('neutral')
        #    head.go()             
        #    self.tries=0 
        #    return'tries'

        
        print('grabbing cent',closest_cent)
        trans_hand, rot_hand = listener.lookupTransform('/hand_palm_link', 'static' + str(closest_cent), rospy.Time(0))
        print(trans_hand)
        wb = whole_body.get_current_joint_values()
        wb[0] += trans_hand[2] - 0.05
        
        wb[1] += trans_hand[1]
        succ = whole_body.go(wb)
        trans_hand, rot_hand = listener.lookupTransform('/hand_palm_link', 'static'+ str(closest_cent), rospy.Time(0))
        move_hand(0)
        if succ:
            return 'succ'
        else:
            arm.set_named_target('go')
            arm.go()
            head.set_named_target('neutral')
            head.go()       
            return 'failed'


##### Define state POST_FLOOR #####
#Se hace para atras, verifica grasp y pone posicion neutral
class Post_floor(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        self.tries+=1
        print(self.tries,'out of 2')
        if self.tries==2:
            self.tries=0 
            arm.set_named_target('go')
            arm.go()
            head.set_named_target('neutral')
            head.go()  

            return'tries'
        wb = whole_body.get_current_joint_values()
        wb[0] += - 0.3
        whole_body.go(wb)
        a = gripper.get_current_joint_values()
        if np.linalg.norm(a - np.asarray(grasped))  >  (np.linalg.norm(a - np.asarray(ungrasped))):
            print ('grasp seems to have failed')
            return 'failed'
        else:
            print('super primitive grasp detector points towards succesfull ')
            arm.set_named_target('go')
            arm.go()
            head.set_named_target('neutral')
            head.go()  
            goal_x, goal_y, goal_yaw =  kl_mess1
            move_base_goal(goal_x, goal_y , -90)
            self.tries=0  
            return 'succ'
            



##### Define state GO_BOX #####
#Se mueve hacia la caja baja el brazo y se acerca mas 
class Go_box(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        self.tries+=1
        print(self.tries,'out of 5')
        if self.tries==5:
            self.tries=0 
            return 'tries'
        goal_x, goal_y, goal_yaw =  kl_tray #Known location tray 1
        a = gripper.get_current_joint_values()
        #if np.linalg.norm(a - np.asarray(grasped)) > (np.linalg.norm(a - np.asarray(ungrasped))):
        #    print ('object might have fallen')
        #    return'tries'


        succ = move_base_goal(goal_x, goal_y+0.35 , -90)
        publish_scene()
        if succ:
            self.tries=0 
            return 'succ'
        return 'failed'
        

##### Define state DELIVER #####
#Suelta el objeto
class Deliver(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        
       
        self.tries+=1
        print(self.tries,'out of 5')
        if self.tries==5:
            self.tries=0 
            return 'tries'
       

        arm.set_joint_value_target(arm_ready_to_place)
        arm.go()

        trans_hand, rot_hand = listener.lookupTransform('Box1','/hand_palm_link', rospy.Time(0))
        print ('hand wrt box1',trans_hand)
        arm.set_joint_value_target(arm_ready_to_place)
        arm.go()
        wb = whole_body.get_current_joint_values()
        wb[0] += -trans_hand[1]
        wb[1] += -trans_hand[0]
        try:
            succ = whole_body.go(wb)
        except MoveItCommanderException:
            return  'failed'
        trans_hand, rot_hand = listener.lookupTransform('Box1','/hand_palm_link', rospy.Time(0))
        wb = whole_body.get_current_joint_values()
        wb[0] += -trans_hand[1]
        wb[1] += -trans_hand[0]
        
        try:
            succ = whole_body.go(wb)
        except MoveItCommanderException:
            return  'failed'
        trans_hand, rot_hand = listener.lookupTransform('Box1','/hand_palm_link', rospy.Time(0))
        print ('hand wrt box1',trans_hand)
        move_hand(1)
        """wb = whole_body.get_current_joint_values()
                                wb[0] += trans_hand[2]
                                wb[1] += trans_hand[1]
                                try:
                                    succ = whole_body.go(wb)
                                except MoveItCommanderException:
                                    return  'failed'
                        
                                
                                succ = whole_body.go(wb)
                                trans_hand, rot_hand = listener.lookupTransform('/hand_palm_link', 'Box1', rospy.Time(0))
                                print ('hand wrt box1',trans_hand)
                                wb = whole_body.get_current_joint_values()
                                wb[0] += -trans_hand[2]
                                wb[1] += -trans_hand[1]
                                try:
                                    succ = whole_body.go(wb)
                                except MoveItCommanderException:
                                    return  'failed'"""
            
        
        move_hand(1)
        
        if succ:
            wb = whole_body.get_current_joint_values()
            wb[0] += 0.45
            whole_body.set_joint_value_target(wb)
            succ=whole_body.go()
            move_hand(0)
            arm.set_named_target('go')
            arm.go()
            head.set_named_target('neutral')
            head.go()
            return 'succ'
        else:
            return 'failed'





#TABLE


##### Define state SCAN_TABLE #####
class Scan_table(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['counter_in'],output_keys=['counter_out'])
        self.tries=0
    def execute(self,userdata):
        self.tries+=1
        if self.tries==3:
            self.tries=0 
            return'tries'
        global cents, rot, trans
        print(userdata.counter_in)
        userdata.counter_out=userdata.counter_in +1

        goal_x , goal_y, goal_yaw = kl_table1
        move_base_goal(goal_x+.25*self.tries, goal_y , goal_yaw)      
        head_val = head.get_current_joint_values()
        head_val[0] = np.deg2rad(0)
        head_val[1] = np.deg2rad(-45)        
        succ = head.go(head_val)
        rospy.sleep(.2)
        
        trans, rot = listener.lookupTransform('/map', '/head_rgbd_sensor_gazebo_frame', rospy.Time(0))
        euler = tf.transformations.euler_from_quaternion(rot)        
        cents = segment_table()
        if len (cents)==0:
            cents = segment_table2(2)
            
            
                                            
        if len (cents)==0:
            arm.set_named_target('go')
            arm.go()
            head.set_named_target('neutral')
            head.go()
            return 'failed'
        else:
            print ('static tfs published')
            static_tf_publish(cents)
            self.tries=0 
            return 'succ'
        
class Scan_table2(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        self.tries+=1
        if self.tries==3:
            self.tries=0 
            return'tries'
    
        global cents, rot, trans
       
        goal_x , goal_y, goal_yaw = kl_table2
        succ = move_base_goal(goal_x+.1*self.tries, goal_y , goal_yaw)      
        head_val = head.get_current_joint_values()
        head_val[0] = np.deg2rad(0)
        head_val[1] = np.deg2rad(-30)        
        head.go(head_val)
        
        trans, rot = listener.lookupTransform('/map', '/head_rgbd_sensor_gazebo_frame', rospy.Time(0))
        euler = tf.transformations.euler_from_quaternion(rot)        
        cents = segment_table2(self.tries)
        if (len (cents) ==0):
            return 'failed'                                          
        else:
            static_tf_publish(cents)
            print("cents  wrt head" +str(cents))
            self.tries=0
            return 'succ'
        


##### Define state PRE_TABLE #####
class Pre_table(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0
    def execute(self,userdata):
        self.tries+=1
        if self.tries==3:
            self.tries=0 
            return'tries'
    
        global closest_cent 
        global cents              

        print("centroids wrt head " +str(cents))
        publish_scene()

        trans_cents = []        
        for i, cent in enumerate(cents):
            trans_map, _ = listener.lookupTransform('/map', 'static' + str(i), rospy.Time(0))
            trans_cents.append(trans_map)
        
        print("centroids wrt map" + str(trans_cents))
       
        if len(trans_cents) !=0:

            np.linalg.norm(np.asarray(trans_cents) - trans , axis = 1)
            closest_cent = np.argmin(np.linalg.norm(np.asarray(trans_cents) - trans , axis = 1))
            print("Closest Cent " + str(closest_cent))
            
        else: 
            print("no object found")
            return 'failed'
        head_val = head.get_current_joint_values()
        head_val[0] = np.deg2rad(0)
        head_val[1] = np.deg2rad(-45)
        head.go(head_val)
        cents = segment_table()
        static_tf_publish(cents)
        publish_scene()
        
        move_hand(1)
        arm.set_joint_value_target(arm_grasp_table)
        succ=arm.go()
        if succ:
            return 'succ'
        else:
            return 'failed'
class Pre_table2(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        self.tries+=1
        if self.tries==3:
            self.tries=0 
            return'tries'
    
        global closest_cent 
        global cents              
    
        trans, rot = listener.lookupTransform('/map', '/head_rgbd_sensor_gazebo_frame', rospy.Time(0))
        euler = tf.transformations.euler_from_quaternion(rot)
        print("centroids wrt head " +str(cents))
        publish_scene()
        cents= segment_table2(2)
        trans_cents = []        
        for i, cent in enumerate(cents):
            trans_map, _ = listener.lookupTransform('/map', 'static' + str(i), rospy.Time(0))
            trans_cents.append(trans_map)
        
        print("centroids wrt map" + str(trans_cents))
        if trans_cents[0]> .3:
            print('not part of table 2')
        if len(trans_cents) !=0:

            np.linalg.norm(np.asarray(trans_cents) - trans , axis = 1)
            closest_cent = np.argmin(np.linalg.norm(np.asarray(trans_cents) - trans , axis = 1))
            print("Closest Cent " + str(closest_cent))
            publish_scene()
            
            move_hand(1)
            arm.set_joint_value_target(arm_grasp_table)
            succ=arm.go()
            if succ:
                return 'succ'
            else:
                return 'failed'

            
        else: 
            print("no object found")
            return 'failed'
        
        
        
        
        
        
                     
        
       

     
#Define state GRASP_TABLE
class Grasp_table(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        self.tries+=1
        if self.tries==5:
            self.tries=0 
            return'tries'
        

        trans_hand, rot_hand = listener.lookupTransform('/hand_palm_link', 'static'+str(closest_cent), rospy.Time(0))        
        wb = whole_body.get_current_joint_values()
        wb[0] += trans_hand[2] - 0.15
        wb[1] += trans_hand[1]
        wb[3] += trans_hand[0]+0.15
        whole_body.go(wb)
        trans_hand, rot_hand = listener.lookupTransform('/hand_palm_link', 'static'+str(closest_cent), rospy.Time(0))
        scene.remove_world_object()
        wb = whole_body.get_current_joint_values()
        wb[0] += trans_hand[2] - 0.06
        wb[1] += trans_hand[1]
        wb[3] += trans_hand[0]+.07
        succ = whole_body.go(wb)
        
        trans_hand, rot_hand = listener.lookupTransform('/hand_palm_link', 'static'+str(closest_cent), rospy.Time(0))
        scene.remove_world_object()
        wb = whole_body.get_current_joint_values()
        wb[0] += trans_hand[2] - 0.06
        wb[1] += trans_hand[1]
        wb[3] += trans_hand[0]
        succ = whole_body.go(wb)
        move_hand(0)
        if succ:
            self.tries=0
            return'succ'
        else:
            return 'failed'


##### Define state POST_TABLE #####
class Post_table(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        self.tries+=1
        if self.tries==5:
            self.tries=0 
            return'tries'
        a = gripper.get_current_joint_values()
        if np.linalg.norm(a - np.asarray(grasped)) > (np.linalg.norm(a - np.asarray(ungrasped))):
            print ('grasp seems to have failed')
            return 'failed'
        else:
            print('assuming succesful grasp')
            wb = whole_body.get_current_joint_values()
            wb[0] += -0.2
            wb[3] += 0.2
            whole_body.set_joint_value_target(wb)
            whole_body.go()
            self.tries=0 
            publish_scene()
            #Takeshi neutral
            arm.set_named_target('go')
            arm.go()
            head.set_named_target('neutral')
            head.go()
            return 'succ'

class Pre_table2(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        self.tries+=1
        if self.tries==3:
            self.tries=0 
            return'tries'
    
        global closest_cent 
        global cents              

        print("centroids wrt head " +str(cents))
        publish_scene()

        trans_cents = []        
        for i, cent in enumerate(cents):
            trans_map, _ = listener.lookupTransform('/map', 'static' + str(i), rospy.Time(0))
            trans_cents.append(trans_map)
        
        print("centroids wrt map" + str(trans_cents))
        if len(trans_cents) !=0:

            np.linalg.norm(np.asarray(trans_cents) - trans , axis = 1)
            closest_cent = np.argmin(np.linalg.norm(np.asarray(trans_cents) - trans , axis = 1))
            print("Closest Cent " + str(closest_cent))
        else: 
            print("no object found")
            return 'failed'
        move_hand(1)
        arm.set_joint_value_target(arm_grasp_table)
        succ=arm.go()
        if succ:
           return 'succ'

        else:
            return 'failed'
        
        
                     
        
       
class Pre_drawer(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        self.tries+=1
        goal_x , goal_y, goal_yaw = kl_drawers 
        ### A KNOWN LOCATION NAMED DRAWERS ( LOCATED utils_takeshi.py)       
        succ = move_base_goal(goal_x, goal_y, goal_yaw) 
        publish_scene()
        ##### TF of the know location drawers will be published in this funtion
        succ=arm.go(arm_grasp_table)
        #### Preknow grasping position called grasp table
        move_hand(1)      
        
        if self.tries==3:
            self.tries=0 
            return'tries'
        if succ:
            return 'succ'
        else:
            return 'failed'
        
class Grasp_drawer(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        publish_scene()
        self.tries+=1
        ## BY COMPARING RELATIVE POSITION BETWEEN HAND AND DRAWER A TWO STAGE SMALL ADJUSTMENTS MOVEMENT IS PROPOSED
        trans_hand,rot_hand= listener.lookupTransform( 'Drawer_high','hand_palm_link',rospy.Time(0))
        wb=whole_body.get_current_joint_values()
        wb[0]+=-trans_hand[1]+.15
        wb[1]+=-trans_hand[0]
        wb[3]+=-trans_hand[2]
        whole_body.go(wb)
        trans_hand,rot_hand= listener.lookupTransform( 'Drawer_high','hand_palm_link',rospy.Time(0))
        wb=whole_body.get_current_joint_values()
        wb[0]+=-trans_hand[1]+.08
        wb[1]+=-trans_hand[0]
        wb[3]+=-trans_hand[2]
        succ=whole_body.go(wb)
        
        move_hand(0)      
        
        if self.tries==3:
            self.tries=0 
            return'tries'
        if succ:
            return 'succ'
        else:
            return 'failed'
        
        
class Post_drawer(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        publish_scene()
        self.tries+=1
        move_hand(0)


        wb=whole_body.get_current_joint_values()
        wb[0]+= 0.3
        whole_body.go(wb)
        
        succ=move_hand(1)
        wb[3]+= 0.3
        rospy.sleep(.5)
        
        
        if self.tries==3:
            self.tries=0 
            return'tries'
        if succ:
            return 'succ'
        else:
            return 'failed'


        


#Initialize global variables and node
def init(node_name):
    global listener, broadcaster, tfBuffer, tf_static_broadcaster, scene, rgbd    
    rospy.init_node(node_name)
    listener = tf.TransformListener()
    broadcaster = tf.TransformBroadcaster()
    tfBuffer = tf2_ros.Buffer()
    tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster()
    whole_body.set_workspace([-6.0, -6.0, 6.0, 6.0]) 
    scene = moveit_commander.PlanningSceneInterface()
    rgbd = RGBD()

     ##FIXING TF TO MAP ( ODOM REALLY)    
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

    
    
  

#Entry point    
if __name__== '__main__':
    print("Takeshi STATE MACHINE...")
    init("takeshi_smach")
    sm = smach.StateMachine(outcomes = ['END'])     #State machine, final state "END"
    sm.userdata.sm_counter = 0

    with sm:
        #State machine for grasping on Floor
        smach.StateMachine.add("INITIAL",       Initial(),      transitions = {'failed':'INITIAL',      'succ':'PRE_DRAWER',    'tries':'INITIAL'}) 
        smach.StateMachine.add("SCAN_FLOOR",    Scan_floor(),   transitions = {'failed':'SCAN_FLOOR',   'succ':'PRE_FLOOR',     'tries':'SCAN_TABLE','change':'SCAN_TABLE2'}) 
        smach.StateMachine.add('PRE_FLOOR',     Pre_floor(),    transitions = {'failed':'PRE_FLOOR',    'succ': 'GRASP_FLOOR',  'tries':'SCAN_TABLE'},remapping={'counter_in':'sm_counter','counter_out':'sm_counter'}) 
        smach.StateMachine.add('PRE_DRAWER',     Pre_drawer(),  transitions = {'failed':'PRE_DRAWER',    'succ': 'GRASP_DRAWER',  'tries':'END'}) 
        smach.StateMachine.add('GRASP_DRAWER',     Grasp_drawer(),  transitions = {'failed':'PRE_DRAWER',    'succ': 'END',  'tries':'INITIAL'}) 
        smach.StateMachine.add('POST_DRAWER',     Post_drawer(),  transitions = {'failed':'GRASP_DRAWER',    'succ': 'INITIAL',  'tries':'END'}) 
        smach.StateMachine.add('GRASP_FLOOR',   Grasp_floor(),  transitions = {'failed':'SCAN_TABLE',  'succ': 'POST_FLOOR',   'tries':'INITIAL'}) 
        smach.StateMachine.add('POST_FLOOR',    Post_floor(),   transitions = {'failed':'GRASP_FLOOR',  'succ': 'GO_BOX',       'tries':'SCAN_FLOOR'}) 
        smach.StateMachine.add('GO_BOX',        Go_box(),       transitions = {'failed':'GO_BOX',       'succ': 'DELIVER',      'tries':'INITIAL'})
        smach.StateMachine.add('DELIVER',       Deliver(),      transitions = {'failed':'DELIVER',      'succ': 'SCAN_FLOOR', 'tries':'GO_BOX'})
        smach.StateMachine.add("SCAN_TABLE",    Scan_table(),   transitions = {'failed':'SCAN_TABLE',   'succ':'PRE_TABLE',     'tries':'SCAN_TABLE2'},remapping={'counter_in':'sm_counter','counter_out':'sm_counter'})
        smach.StateMachine.add("SCAN_TABLE2",   Scan_table2(),  transitions = {'failed':'SCAN_TABLE2',   'succ':'PRE_TABLE2',     'tries':'INITIAL'}) 
        smach.StateMachine.add('PRE_TABLE',     Pre_table(),    transitions = {'failed':'PRE_TABLE',    'succ': 'GRASP_TABLE',  'tries':'SCAN_TABLE2'}) 
        smach.StateMachine.add('GRASP_TABLE',   Grasp_table(),  transitions = {'failed':'GRASP_TABLE',  'succ': 'POST_TABLE',   'tries':'SCAN_TABLE2'}) 
        smach.StateMachine.add('POST_TABLE',    Post_table(),   transitions = {'failed':'PRE_TABLE2',  'succ': 'GO_BOX',       'tries':'INITIAL'}) 
        smach.StateMachine.add('PRE_TABLE2',    Pre_table2(),   transitions = {'failed':'PRE_TABLE2',    'succ': 'GRASP_TABLE',  'tries':'INITIAL'}) 
        

        

      

    outcome = sm.execute()


    