#!/usr/bin/env python

from utils_takeshi import *
import datetime

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
def add_transform(child, trans, rot, parent="map"):
    static_transformStamped = TransformStamped()
    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = parent
    static_transformStamped.child_frame_id = child
    static_transformStamped.transform.translation.x = trans[0]
    static_transformStamped.transform.translation.y = trans[1]
    static_transformStamped.transform.translation.z = trans[2]
    static_transformStamped.transform.rotation.x = rot[0]    
    static_transformStamped.transform.rotation.y = rot[1]    
    static_transformStamped.transform.rotation.z = rot[2]    
    static_transformStamped.transform.rotation.w = rot[3]    
    tf_static_broadcaster.sendTransform(static_transformStamped)

def lineup_table():
    
    cv2_img=rgbd.get_image()
    img=cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    img=cv2.Canny(img,80,200)
    lines = cv2.HoughLines(img,1,np.pi/180,150)
    
   
    
    wb=whole_body.get_current_joint_values()
    wb[2]+=-(lines[0,0,1]-.5*np.pi)
    if (np.isclose(lines[0,0,1], [1.57], atol=0.3)):
        succ=whole_body.go(wb)
    
    lines=np.asarray(lines)
    l=len(lines)
    lines=lines.ravel().reshape(l,2)
    table_limit_px=[]

    for line in lines :
        if (np.isclose(line[1], [1.57], atol=0.1)):
            table_limit_px.append(line[0])
    if len (lines)<2: 
        table_region=np.asarray([100, 200])
        print('table region failed')
    else:
        table_region=np.asarray([np.min(table_limit_px), np.max(table_limit_px)])
        print (table_region)
        
    return table_region 

        
def rot_to_euler(R):
    import sys
    tol = sys.float_info.epsilon * 10

    if abs(R.item(0,0))< tol and abs(R.item(1,0)) < tol:
       eul1 = 0
       eul2 = m.atan2(-R.item(2,0), R.item(0,0))
       eul3 = m.atan2(-R.item(1,2), R.item(1,1))
    else:   
       eul1 = m.atan2(R.item(1,0),R.item(0,0))
       sp = m.sin(eul1)
       cp = m.cos(eul1)
       eul2 = m.atan2(-R.item(2,0),cp*R.item(0,0)+sp*R.item(1,0))
       eul3 = m.atan2(sp*R.item(0,2)-cp*R.item(1,2),cp*R.item(1,1)-sp*R.item(0,1))

    return np.asarray((eul1,eul2,eul3))
#def pca_xyz(xyz):
#    quats=[]
#    for i in range( len(xyz)):
#        pca= PCA(n_components=3).fit(xyz[i])
#        vec0= pca.components_[0,:]
#        vec1= pca.components_[1,:]
#        vec2= pca.components_[2,:]
#        R=pca.components_
#        euler=rot_to_euler(R)
#        quats.append(tf.transformations.quaternion_from_euler(euler[0],euler[1],euler[2]))
#    return quats


def seg_pca(lower=2000,higher=50000,reg_ly=0,reg_hy=1000): 
    image= rgbd.get_h_image()
    points_data= rgbd.get_points()
    values=image.reshape((-1,3))
    values= np.float32(values)
    criteria= (  cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER  ,1000,0.1)
    k=6
    _ , labels , cc =cv2.kmeans(values , k ,None,criteria,30,cv2.KMEANS_RANDOM_CENTERS)
    cc=np.uint8(cc)
    segmented_image= cc[labels.flatten()]
    segmented_image=segmented_image.reshape(image.shape)
    th3 = cv2.adaptiveThreshold(segmented_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    kernel = np.ones((5,5),np.uint8)
    im4=cv2.erode(th3,kernel,iterations=4)
    plane_mask=points_data['z']
    cv2_img=plane_mask.astype('uint8')
    img=im4
    _,contours, hierarchy = cv2.findContours(im4.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    i=0
    cents=[]
    points=[]
    for i, contour in enumerate(contours):
        
        area = cv2.contourArea(contour)

        if area > lower and area < higher :
            M = cv2.moments(contour)
            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
    
            boundRect = cv2.boundingRect(contour)
            #just for drawing rect, dont waste too much time on this

            img=cv2.rectangle(img,(boundRect[0], boundRect[1]),(boundRect[0]+boundRect[2], boundRect[1]+boundRect[3]), (0,0,0), 2)
            # calculate moments for each contour
            if (cY > reg_ly and cY < reg_hy  ):
                
                cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
                cv2.putText(img, "centroid_"+str(i)+"_"+str(cX)+','+str(cY)    ,    (cX - 25, cY - 25)   ,cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                print ('cX,cY',cX,cY)
                xyz=[]


                for jy in range (boundRect[0], boundRect[0]+boundRect[2]):
                    for ix in range(boundRect[1], boundRect[1]+boundRect[3]):
                        aux=(np.asarray((points_data['x'][ix,jy],points_data['y'][ix,jy],points_data['z'][ix,jy])))
                        if np.isnan(aux[0]) or np.isnan(aux[1]) or np.isnan(aux[2]):
                            'reject point'
                        else:
                            xyz.append(aux)

                xyz=np.asarray(xyz)
                cent=xyz.mean(axis=0)
                cents.append(cent)
                print (cent)
                points.append(xyz)
            else:
                print ('cent out of region... rejected')
            
    cents=np.asarray(cents)
    ### returns centroids found and a group of 3d coordinates that conform the centroid
    return(cents,np.asarray(points))



        
def seg_floor(image_data,points_data,lower=200,upper=50000):
   # image_data=rgbd.get_image()
   # points_data = rgbd.get_points()


    ##px pixels /2D world  P1 3D world
    px_y,px_x=-1,-200
    P1= np.asarray((points_data[px_y,px_x]['x'],points_data[px_y,px_x]['y'],points_data[px_y,px_x]['z'] ))
    px_y,px_x=-1,200
    P2= np.asarray((points_data[px_y,px_x]['x'],points_data[px_y,px_x]['y'],points_data[px_y,px_x]['z'] ))
    px_y,px_x=-150,320
    P3= np.asarray((points_data[px_y,px_x]['x'],points_data[px_y,px_x]['y'],points_data[px_y,px_x]['z'] ))
    #      

    V1 =P1 - P2
    V2= P3-P2
    nx,ny,nz=np.cross(V2,V1)
    print('look at the phi angle  in normal vector', np.rad2deg(cart2spher(nx,ny,nz))[2]-90)
    trans , rot = listener.lookupTransform('/map', '/head_rgbd_sensor_gazebo_frame', rospy.Time(0))
    euler=tf.transformations.euler_from_quaternion(rot)
    print(   np.rad2deg(euler)[1],'if this degree is not the same as head tilt plane was not found')
    
    mask=np.zeros((image_data.shape))
    plane_mask=np.zeros((image_data.shape[0],image_data.shape[1]))
    mask[:,:,0]=points_data['x'] - P1[0]
    mask[:,:,1]=points_data['y'] - P1[1]
    mask[:,:,2]=points_data['z'] - P1[2]
    for i in range (image_data.shape[0]):
        for j in range (image_data.shape[1]):
            plane_mask[i,j]=-np.dot(np.asarray((nx,ny,nz,)),mask[i,j])
    plane_mask=plane_mask-np.min(plane_mask)
    plane_mask=plane_mask*256/np.max(plane_mask)
    plane_mask.astype('uint8')

    ret,thresh = cv2.threshold(plane_mask,3,255,0)

    cv2_img=plane_mask.astype('uint8')
    img=plane_mask.astype('uint8')
    _,contours, hierarchy = cv2.findContours(thresh.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    i=0
    cents=[]
    points=[]
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        if area > lower and area < upper :
            #print('contour',i,'area',area)
            
            boundRect = cv2.boundingRect(contour)
            #just for drawing rect, dont waste too much time on this
            img=cv2.rectangle(img,(boundRect[0], boundRect[1]),(boundRect[0]+boundRect[2], boundRect[1]+boundRect[3]), (255,0,0), 2)
            # calculate moments for each contour
            xyz=[]
            
            
            for jy in range (boundRect[0], boundRect[0]+boundRect[2]):
                for ix in range(boundRect[1], boundRect[1]+boundRect[3]):
                    aux=(np.asarray((points_data['x'][ix,jy],points_data['y'][ix,jy],points_data['z'][ix,jy])))
                    if np.isnan(aux[0]) or np.isnan(aux[1]) or np.isnan(aux[2]):
                        'reject point'
                    else:
                        xyz.append(aux)
                        
            xyz=np.asarray(xyz)
            cent=xyz.mean(axis=0)
            cents.append(cent)
            points.append(xyz)
            #M = cv2.moments(contour)
            # calculate x,y coordinate of center
            #cX = int(M["m10"] / M["m00"])
            #cY = int(M["m01"] / M["m00"])
            #cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
            #cv2.putText(img, "centroid_"+str(i)+"_"+str(cX)+','+str(cY)    ,    (cX - 25, cY - 25)   ,cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cents=np.asarray(cents)
    #plt.imshow(img)
    return(cents,np.asarray(points))


##DEPRECATION ALERT 
"""def segment_floor():
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
"""

#TF WRT HEAD SENSOR
def static_tf_publish(cents, quaternions=[]):
    if (len(quaternions))==0:
        quats=np.zeros((len(cents),4)) 
        quats[:,3]=1
        #print quats
    else:
        quats=np.asarray(quaternions)
        #print quats
    for  i ,cent  in enumerate(cents):
        x,y,z=cent
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            print('nan , rejected')
        else:
            #### first place a dissolving tf wrt head sensor  in centroids
            broadcaster.sendTransform((x,y,z),rot, rospy.Time.now(), 'Closest_Object'+str(i),"head_rgbd_sensor_link")
            rospy.sleep(.2)
            
            #### then place each centr wrt map
            xyz_map,cent_quat= listener.lookupTransform('/map', 'Closest_Object'+str(i),rospy.Time(0))
            map_euler=tf.transformations.euler_from_quaternion(cent_quat)
            rospy.sleep(.2)
            static_transformStamped = TransformStamped()

            ##FIXING TF TO MAP ( ODOM REALLY)    
            #tf_broadcaster1.sendTransform( (xyz[0],xyz[1],xyz[2]),tf.transformations.quaternion_from_euler(0, 0, 0), rospy.Time.now(), "obj"+str(ind), "head_rgbd_sensor_link")
            ## Finally boiradcast a static tf  in cents and with quaternion found  in pca
            static_transformStamped.header.stamp = rospy.Time.now()
            static_transformStamped.header.frame_id = "map"
            static_transformStamped.child_frame_id = "static"+str(i)
            static_transformStamped.transform.translation.x = float(xyz_map[0])
            static_transformStamped.transform.translation.y = float(xyz_map[1])
            static_transformStamped.transform.translation.z = float(xyz_map[2])
            #quat = tf.transformations.quaternion_from_euler(-euler[0],0,1.5)
            static_transformStamped.transform.rotation.x = quats [i,0]#-quat[0]#trans.transform.rotation.x
            static_transformStamped.transform.rotation.y = quats [i,1]#-quat[1]#trans.transform.rotation.y
            static_transformStamped.transform.rotation.z = quats [i,2]#-quat[2]#trans.transform.rotation.z
            static_transformStamped.transform.rotation.w = quats [i,3]#-quat[3]#trans.transform.rotation.w


            tf_static_broadcaster.sendTransform(static_transformStamped)
    return True





def add_object(name, size, pose, orientation):
    p = PoseStamped()
    p.header.frame_id = "map"       # "head_rgbd_sensor_link"
    
    p.pose.position.x = pose[0]
    p.pose.position.y = pose[1]
    p.pose.position.z = pose[2]

    p.pose.orientation.x = orientation[0] * np.pi
    p.pose.orientation.y = orientation[1] * np.pi
    p.pose.orientation.z = orientation[2] * np.pi
    p.pose.orientation.w = orientation[3] * np.pi

    scene.add_box(name, p, size)


def publish_scene():
    add_object("shelf", [1.5, 0.04, 0.4],  [2.5, 4.85, 0.78],  [0.5,0,0,0.5])
    add_object("shelf1", [1.5, 0.04, 0.4], [2.5, 4.85, 0.49], [0.5,0,0, 0.5])
    add_object("shelf2", [1.5, 0.04, 0.4], [2.5, 4.85, 0.18], [0.5,0,0, 0.5])
    add_object("shelf_wall", [1, 1, 0.04], [2.5, 4.9, 0.5], [0.5,0,0, 0.5])
    add_object("shelf_wall1", [.04, 1, 0.4], [2.7, 4.9, 0.5],[0.5,0,0, 0.5])
    add_object("shelf_wall2", [.04, 1, 0.4], [1.8, 4.9, 0.5], [0.5,0,0 ,0.5])    
    add_object("table_big", [1.5, 0.13, 0.5], [0.95, 1.9, 0.34],  [0.5,0,0, 0.5])
    add_object("table_big_legs1",[.01,.6,.2], [1.55,1.8,0.1],       [0.5,0,0, 0.5])
    add_object("table_big_legs2",[.01,.6,.2], [0.45,1.8,0.1],       [0.5,0,0 ,0.5])
    add_object("table_small", [0.9, 0.02, 0.4], [-0.2, 1.85, 0.61],  [0.5,0,0 ,0.5])
    add_object("table_small_legs1",[.01,.6,.2], [-0.3,1.75,0.3],      [0.5,0,0, 0.5])
    add_object("table_small_legs2",[.01,.6,.2], [0.1,1.75,0.3], [0.5,0,0 ,0.5])
    add_object("table_tray", [0.65, 0.01, 0.7], [1.8, -0.65, 0.4], [0.5,0,0, 0.5])
    add_object("containers", [0.3, 0.3, 0.3], [1.4, -0.65, 0.4], [0.5,0,0, 0.5])
    add_object("drawers", [1, 1, 1], [0, -0.65, 0.5], [0.5,0,0, 0.5])

    add_object("big_wall" , [6.0, 0.2, 0.2], [3.2,  2.0, 0.0],  [0,0.0,0.5 ,0.5])
    add_object("mid_wall" , [4.0, 0.2, 0.2], [0.1,  2.1, 0.0],  [0,0.0,0.0 ,1/np.pi])
    add_object("door_wall" , [5.0, 0.2, 0.2], [-0.8, 2.8, 0.0],  [0,0.0,0.5 ,0.5     ])
    add_object("close_wall", [4.0, 0.2, 0.2], [1.1, -0.5, 0.0],  [0,0.0,0.0 ,1/np.pi])
    add_object("far_wall",   [4.0, 0.2, 0.2], [1.1, 5.0, 0.0],  [0,0.0,0.0 ,1/np.pi])
    
    add_transform("Tray_A", [1.665, -0.59, 0.5], [0, 0, 0, 1])
    add_transform("Tray_B", [1.97, -0.59, 0.5], [0, 0, 0, 1])

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
    static_transformStamped.transform.translation.x = 2.3
    static_transformStamped.transform.translation.y = -0.5
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




        rospy.loginfo('STATE : Initialization')
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




class Scan_floor(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries','change'],input_keys=['timer_in'],output_keys=['timer_out'])
        self.tries=0
    def execute(self,userdata):

        #self.tries+=1
        
        print(self.tries,'of 3')
        if self.tries==3:
            self.tries=0 
            print('lets try table now')
            return'tries'
        global cents, rot, trans , end_time
        goal_x , goal_y, goal_yaw = kl_mess1        
        now= datetime.datetime.now()
        if userdata.timer_in:
            end_time=now + datetime.timedelta(minutes = 5)
            userdata.timer_out=False
            rospy.loginfo('Started moving at '+ str (now))
        print ('Time ', now, 'will end at ', end_time)
        succ = move_base_goal(goal_x, goal_y, goal_yaw)        
        head_val = head.get_current_joint_values()
        head_val[0] = np.deg2rad(0)
        head_val[1] = np.deg2rad(-45)        
        head.go(head_val)
        trans, rot = listener.lookupTransform('/map', '/head_rgbd_sensor_gazebo_frame', rospy.Time(0))
        euler = tf.transformations.euler_from_quaternion(rot)
        #print(trans, euler)
        #cents = segment_floor()
        cents,xyz=seg_floor(rgbd.get_image(),rgbd.get_points())
        #cents, xyz = seg_pca()


        if len (cents!=0):
            #quats=pca_xyz(xyz)
            static_tf_publish(cents)
            self.tries=0 
            return 'succ'

        #cents_to_sceneobjs(cents) 

        return 'failed'
            

##### Define state PRE_FLOOR #####
#Baja el brazo al suelo, abre la garra y se acerca al objeto para grasp
class Pre_floor(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['counter_in'],output_keys=['counter_out'])
        self.tries=0
    def execute(self,userdata):
        now= datetime.datetime.now()
        if now > end_time:
            print ('5 mins up')
            rospy.loginfo('5 mins up')
            return  0
            
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
            print ("YOOOLOOOOOOOOOOO (not the algorithm)")
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
        trans_hand, rot_hand = listener.lookupTransform('/hand_palm_link', 'static' + str(closest_cent), rospy.Time(0))
        wb = whole_body.get_current_joint_values()
        wb[0] += trans_hand[2] -0.05
        wb[1] += trans_hand[1]
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
        now= datetime.datetime.now()
        if now > end_time:
            print ('5 mins up')
            rospy.loginfo('5 mins up')
            return  0
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
        now= datetime.datetime.now()
        if now > end_time:
            print ('5 mins up')
            rospy.loginfo('5 mins up')
            return  0
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
            
            self.tries=0  
            return 'succ'
            



##### Define state PRE_DELIVER #####
#Se mueve hacia la caja baja el brazo y se acerca mas 
class Pre_deliver(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['delivery_destination'],output_keys=['delivery_destination_out'])
        self.tries=0
    def execute(self,userdata):
        now= datetime.datetime.now()
        if now > end_time:
            print ('5 mins up')
            rospy.loginfo('5 mins up')
            return  0
        publish_scene()
        print userdata.delivery_destination
        if  (userdata.delivery_destination)=="Tray_A":
            wb=wb_place_tray_A

        if  (userdata.delivery_destination)=="Tray_B":
            wb=wb_place_tray_B

        if  (userdata.delivery_destination)=="Box1":
            wb=wb_place_Box1   

        self.tries+=1
        print(self.tries,'out of 5')
        if self.tries==5:
            self.tries=0 
            return 'tries'
        #goal_x, goal_y, goal_yaw =  kl_tray #Known location tray 1
        

        if self.tries >=3:
            move_base_goal(wb[1]+0.35,wb[0] ,np.rad2deg(wb[2]))
            succ =arm.go(wb[3:])
        else:
            succ=whole_body.go(wb)
            

        #succ = move_base_goal(goal_x, goal_y+0.35 , -90)
        if succ:
            self.tries=0 
            return 'succ'
        
        
        return 'failed'
        

##### Define state DELIVER #####
#Suelta el objeto
class Deliver(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['delivery_destination'],output_keys=['delivery_destination_out'])
        self.tries=0
    def execute(self,userdata):
        now= datetime.datetime.now()
        if now > end_time:
            print ('5 mins up')
            rospy.loginfo('5 mins up')
            return  0
        
        print ("delivering to ",userdata.delivery_destination)       

        self.tries+=1
        print(self.tries,'out of 5')
        if self.tries==5:
            self.tries=0 
            return 'tries'
        """
                             trans_hand,rot_hand= listener.lookupTransform('/hand_palm_link', 'Tray_A',rospy.Time(0))
                             hand_euler=tf.transformations.euler_from_quaternion(rot_hand)
                             print(trans_hand ,hand_euler)
                             wb= whole_body.get_current_joint_values()
                             wb[0]+= -trans_hand[2]
                             wb[1]+= -trans_hand[1]
                     
                             wb[3]+= trans_hand[0]+.05
                             whole_body.go(wb)
        """
        trans_hand,rot_hand= listener.lookupTransform('/hand_palm_link', userdata.delivery_destination ,rospy.Time(0))
        hand_euler=tf.transformations.euler_from_quaternion(rot_hand)
        print(trans_hand ,hand_euler)
        wb= whole_body.get_current_joint_values()
        wb[0]+= -trans_hand[2]
        wb[1]+= -trans_hand[1]

        wb[3]+= trans_hand[0]+.05
        succ=whole_body.go(wb)
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
            if userdata.delivery_destination=='Tray_A':
                userdata.delivery_destination_out= 'Tray_B'
            if userdata.delivery_destination=='Tray_B':
                userdata.delivery_destination_out= 'Box1'

            return 'succ'
        else:
            return 'failed'
    """   arm.set_joint_value_target(arm_ready_to_place)
                       arm.go()
               
                       trans_hand, rot_hand = listener.lookupTransform('Box1','/hand_palm_link', rospy.Time(0))
                       print ('hand wrt box1',trans_hand)
                       arm.set_joint_value_target(arm_ready_to_place)
                       arm.go()
                       wb = whole_body.get_current_joint_values()
                       wb[0] += -trans_hand[1]
                       wb[1] += -trans_hand[0]
                       succ=whole_body.go(wb)
                       trans_hand, rot_hand = listener.lookupTransform('Box1','/hand_palm_link', rospy.Time(0))
                       print ('hand wrt box1',trans_hand)
                       move_hand(1)
                   
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
               
    """



#TABLE


##### Define state SCAN_TABLE #####
class Scan_table(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['counter_in'],output_keys=['counter_out'])
        self.tries=0
    def execute(self,userdata):
        now= datetime.datetime.now()
        if now > end_time:
            print ('5 mins up')
            rospy.loginfo('5 mins up')
            return  0
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
        

        


##### Define state PRE_TABLE #####
class Pre_table(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'],input_keys=['global_counter'])
        self.tries=0
    def execute(self,userdata):
        now= datetime.datetime.now()
        if now > end_time:
            print ('5 mins up')
            rospy.loginfo('5 mins up')
            return  0
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

        
        
        
                     
        
       

     
#Define state GRASP_TABLE
class Grasp_table(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        now= datetime.datetime.now()
        if now > end_time:
            print ('5 mins up')
            rospy.loginfo('5 mins up')
            return  0
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
        now= datetime.datetime.now()
        if now > end_time:
            print ('5 mins up')
            rospy.loginfo('5 mins up')
            return  0
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

class Scan_table2(smach.State):

    def __init__(self):
        smach.State.__init__(self, outcomes = ['succ', 'failed', 'tries'])
        self.tries = 0

    def execute(self, userdata):
        now= datetime.datetime.now()
        if now > end_time:
            print ('5 mins up')
            rospy.loginfo('5 mins up')
            return  0
        self.tries += 1
        if self.tries == 3:
            self.tries = 0 
            return'tries'
    
        
        global cents, rot, trans,  quats , closest_cent
        publish_scene()
        
        arm.set_named_target('go')
        arm.go()
        head.set_named_target('neutral')
        head.go()
        goal_x , goal_y, goal_yaw = kl_table2
        succ = move_base_goal(goal_x-.2+.1*self.tries, goal_y-.1 , goal_yaw+90)      

        head.set_named_target('neutral')
        head.go()

        arm.set_joint_value_target(arm_grasp_table)
        arm.go()
        wb= whole_body.get_current_joint_values()
        wb[3]+=.1
        succ=whole_body.go(wb)


        head_val=head.get_current_joint_values()
        head_val[0]=np.deg2rad(-90)
        head_val[1]=np.deg2rad(-45)
        head.go(head_val)

        table_region=lineup_table()
    
        aux=0
        while abs(table_region[1]-table_region[0]) < 100 :
            aux+=1
            if aux== 5:
                break
                #return 'failed'
            table_region=lineup_table()
            


        publish_scene()
        trans , rot = listener.lookupTransform('/map', '/head_rgbd_sensor_gazebo_frame', rospy.Time(0))
        cents, xyz = seg_pca(2000,30000,table_region[0],table_region[1])
        #quats=pca_xyz(xyz)
        static_tf_publish(cents)
        trans_cents=[]
        for i, cent in enumerate(cents):
            try:
                trans_map, _ = listener.lookupTransform('/map', 'static' + str(i), rospy.Time(0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                
                continue

            trans_cents.append(trans_map)

        print("centroids wrt map" + str(trans_cents))
        if len(trans_cents) !=0:

            np.linalg.norm(np.asarray(trans_cents) - trans , axis = 1)
            closest_cent = np.argmin(np.linalg.norm(np.asarray(trans_cents) - trans , axis = 1))
            print("Closest Cent " + str(closest_cent))
            return 'succ'
        else:
            closest_cent=0
            return 'failed'

class Pre_table2(smach.State):
    
    def __init__(self):
        smach.State.__init__(self, outcomes = ['succ', 'failed', 'tries'])
        self.tries = 0

    def execute(self, userdata):
        now= datetime.datetime.now()
        if now > end_time:
            print ('5 mins up')
            rospy.loginfo('5 mins up')
            return  0
        global grasp_above
        self.tries += 1
        if self.tries == 3:
            self.tries = 0 
            return'tries'
                
        goal_x , goal_y, goal_yaw = kl_table2
        publish_scene()
        #a = quats[closest_cent]
        grasp_above=False
        #np.rad2deg(tf.transformations.euler_from_quaternion(a))
        #grasp_above=False
        #if (np.abs(np.rad2deg(tf.transformations.euler_from_quaternion(a))[1]) < 1):
        #    grasp_above=True
        #    print ("grasp above recommended")
        ##PREGRASPS
        ####
        move_hand(1)

        if  (grasp_above!=True):#or(np.asarray(cents[closest_cent])[2]>0.9):
            print 'TABLE PREGRASP'
            grasp_above=False
            arm.set_named_target('go')
            arm.go()
            head.set_named_target('neutral')
            head.go()
            move_hand(1)
            wb= whole_body.get_current_joint_values()
            wb[0]=goal_y-0.5
            wb[1]=goal_x
            wb[2]=np.deg2rad(goal_yaw)
            wb[3:]=arm_grasp_table
            wb[3] = .6
            succ=whole_body.go(wb)
        else:
            print ' ABOVE PREGRASP'
            whole_body.go(wb_pre_table2_above)
            head.set_named_target('neutral')
            head.go()
            move_hand(1)
            wb= whole_body.get_current_joint_values()
            wb[0]=goal_y-0.35
            wb[1]=goal_x
            wb[2]=np.deg2rad(goal_yaw)
            wb[3] = .66
            wb[7] = 0

            succ=whole_body.go(wb)
        i=closest_cent
        trans_hand,rot_hand= listener.lookupTransform('/hand_palm_link', 'static'+str(i),rospy.Time(0))
        hand_euler=tf.transformations.euler_from_quaternion(rot_hand)
        print (trans_hand)
        if succ:
           return 'succ'
        else:
            return 'failed'

class Grasp_table2(smach.State):

    def __init__(self):
        smach.State.__init__(self, outcomes = ['succ', 'failed', 'tries'])
        self.tries = 0

    def execute(self, userdata):
        now= datetime.datetime.now()
        if now > end_time:
            print ('5 mins up')
            rospy.loginfo('5 mins up')
            return  0
        self.tries += 1
        if self.tries == 5:
            self.tries = 0 
            return'tries'
        #scene.remove_world_object()
        move_hand(1)
        i=closest_cent
        trans_hand,rot_hand= listener.lookupTransform('/hand_palm_link', 'static'+str(i),rospy.Time(0))
        hand_euler=tf.transformations.euler_from_quaternion(rot_hand)
        print(trans_hand )
        if grasp_above:    
            wb= whole_body.get_current_joint_values()
            wb[0]  =  np.min( (1.3, trans_hand[0]+wb[0] ))
            wb[1] += trans_hand[1] 
            wb[3] += -trans_hand[2]+.1
            

            succ=whole_body.go(wb)
            
            #move_hand(1)
        else:

            wb= whole_body.get_current_joint_values()
            wb[0] += trans_hand[2] - 0.3
            wb[1] += trans_hand[1]
            wb[3] =  0.4#trans_hand[0]

            succ=whole_body.go(wb)

            
        trans_hand,rot_hand= listener.lookupTransform('/hand_palm_link', 'static'+str(i),rospy.Time(0))
        hand_euler=tf.transformations.euler_from_quaternion(rot_hand)
        move_hand(1)

        ##GRASP 
        i=closest_cent
        if grasp_above: 
            trans_hand,rot_hand= listener.lookupTransform('/map', 'static'+str(i),rospy.Time(0))
            hand_euler=tf.transformations.euler_from_quaternion(rot_hand)
            print(trans_hand ,hand_euler)

            trans_hand,np.rad2deg(hand_euler)
            print(hand_euler[2])
            a=arm.get_current_joint_values()
            a[-2]=np.max((-0.5*np.pi , hand_euler[2] +.25*np.pi))
            arm.go(a)
            print (a)
            a=arm.get_current_joint_values()
            a[0] =.54
            arm.go(a)
            
            wb=whole_body.get_current_joint_values()
            wb[1]-=.04
            succ=whole_body.go(wb)
            

           
            
            #move_hand(1)
        else:
            trans_hand,rot_hand= listener.lookupTransform('/hand_palm_link', 'static'+str(i),rospy.Time(0))
            hand_euler=tf.transformations.euler_from_quaternion(rot_hand)
            print(trans_hand ,hand_euler)

            #scene.remove_world_object()
            wb= whole_body.get_current_joint_values()
            wb[0] += trans_hand[2] - 0.06
            wb[1] += trans_hand[1]
            wb[3] =  0.36#trans_hand[0]
            succ=whole_body.go(wb)
            scene.remove_world_object()
            

            
        
          
        trans_hand,rot_hand= listener.lookupTransform('/hand_palm_link', 'static'+str(i),rospy.Time(0))
        hand_euler=tf.transformations.euler_from_quaternion(rot_hand)
        print(trans_hand ,hand_euler,succ,wb[0])

        print(trans_hand ,succ,wb[0])
        if succ:
            self.tries = 0
            return 'succ'
        else:
            return 'failed'


class Post_table2(smach.State):

    def __init__(self):
        smach.State.__init__(self, outcomes = ['succ', 'failed', 'tries'])
        self.tries = 0
    def execute(self, userdata):
        now= datetime.datetime.now()
        if now > end_time:
            print ('5 mins up')
            rospy.loginfo('5 mins up')
            return  0
        self.tries += 1
        if self.tries == 5:
            self.tries = 0 
            return'tries'
        scene.remove_world_object()
        a=arm.get_current_joint_values()
        if grasp_above:
            a[0]=.54
            arm.go(a)
        move_hand(0)
        a=arm.get_current_joint_values()
        a[0]=.6
        a[1]=-0.4*np.pi
        arm.go(a)
        g = gripper.get_current_joint_values()
        print g
        if np.linalg.norm(g - np.asarray(grasped)) > (np.linalg.norm(g - np.asarray(ungrasped))):
            print ('grasp seems to have failed')
            return 'failed'
        else:
            print('assuming succesful grasp')
            publish_scene()
            
            
            arm.set_named_target('go')
            arm.go()
            head.set_named_target('neutral')
            head.go()
            self.tries=0
            return 'succ'
        
                     
        
       
class Pre_drawer(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['succ','failed','tries'])
        self.tries=0
    def execute(self,userdata):
        now= datetime.datetime.now()
        if now > end_time:
            print ('5 mins up')
            rospy.loginfo('5 mins up')
            return  0
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
        now= datetime.datetime.now()
        if now > end_time:
            print ('5 mins up')
            rospy.loginfo('5 mins up')
            return  0
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
        now= datetime.datetime.now()
        if now > end_time:
            print ('5 mins up')
            rospy.loginfo('5 mins up')
            return  0
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
    sm.userdata.sm_timer = True
    sm.userdata.delivery_dest = "Tray_A"
    with sm:
        ## SMACH WITH NO DRAWER GRASPING ( UNCOMMMENT LINE FOR DRAWER)
        rospy.loginfo('NO DRAWER')
        #State machine for grasping on Floor
        
        smach.StateMachine.add("INITIAL",       Initial(),      transitions = {'failed':'INITIAL',      'succ':'SCAN_FLOOR',    'tries':'INITIAL'}) 
        smach.StateMachine.add("SCAN_FLOOR",    Scan_floor(),   transitions = {'failed':'SCAN_FLOOR',   'succ':'PRE_FLOOR',     'tries':'SCAN_TABLE','change':'SCAN_TABLE2'},remapping={'timer_in':'sm_timer','timer_out':'sm_timer'}) 
        smach.StateMachine.add('PRE_FLOOR',     Pre_floor(),    transitions = {'failed':'PRE_FLOOR',    'succ': 'GRASP_FLOOR',  'tries':'SCAN_TABLE'},remapping={'counter_in':'sm_counter','counter_out':'sm_counter'}) 
        smach.StateMachine.add('GRASP_FLOOR',   Grasp_floor(),  transitions = {'failed':'SCAN_FLOOR',  'succ': 'POST_FLOOR',   'tries':'INITIAL'}) 
        smach.StateMachine.add('POST_FLOOR',    Post_floor(),   transitions = {'failed':'GRASP_FLOOR',  'succ': 'PRE_DELIVER',       'tries':'SCAN_FLOOR'}) 
        smach.StateMachine.add('PRE_DELIVER',   Pre_deliver(),  transitions = {'failed':'PRE_DELIVER',       'succ': 'DELIVER',      'tries':'INITIAL'},remapping={'delivery_destination':'delivery_dest','delivery_destination_out':'delivery_dest'})
        smach.StateMachine.add('DELIVER',       Deliver(),      transitions = {'failed':'DELIVER',      'succ': 'SCAN_FLOOR', 'tries':'PRE_DELIVER'},remapping={'delivery_destination':'delivery_dest','delivery_destination_out':'delivery_dest'})
        smach.StateMachine.add("SCAN_TABLE",    Scan_table(),   transitions = {'failed':'SCAN_TABLE',   'succ':'PRE_TABLE',     'tries':'SCAN_TABLE2'},remapping={'counter_in':'sm_counter','counter_out':'sm_counter'})
        smach.StateMachine.add('PRE_TABLE',     Pre_table(),    transitions = {'failed':'PRE_TABLE',    'succ': 'GRASP_TABLE',  'tries':'SCAN_TABLE2'}) 
        smach.StateMachine.add('GRASP_TABLE',   Grasp_table(),  transitions = {'failed':'GRASP_TABLE',  'succ': 'POST_TABLE',   'tries':'SCAN_TABLE2'}) 
        smach.StateMachine.add('POST_TABLE',    Post_table(),   transitions = {'failed':'PRE_TABLE2',  'succ': 'PRE_DELIVER',       'tries':'INITIAL'}) 
        smach.StateMachine.add("SCAN_TABLE2",   Scan_table2(),  transitions = {'failed':'SCAN_FLOOR',   'succ':'PRE_TABLE2',     'tries':'INITIAL'}) 
        smach.StateMachine.add('PRE_TABLE2',    Pre_table2(),   transitions = {'failed':'PRE_TABLE2',    'succ': 'GRASP_TABLE2',  'tries':'INITIAL'}) 
        smach.StateMachine.add('GRASP_TABLE2',   Grasp_table2(),  transitions = {'failed':'GRASP_TABLE2',  'succ': 'POST_TABLE2',   'tries':'SCAN_TABLE2'}) 
        smach.StateMachine.add('POST_TABLE2',    Post_table2(),   transitions = {'failed':'PRE_TABLE2',  'succ': 'PRE_DELIVER',       'tries':'INITIAL'}) 
        smach.StateMachine.add('PRE_DRAWER',     Pre_drawer(),  transitions = {'failed':'PRE_DRAWER',    'succ': 'GRASP_DRAWER',  'tries':'END'}) 
        smach.StateMachine.add('GRASP_DRAWER',     Grasp_drawer(),  transitions = {'failed':'PRE_DRAWER',    'succ': 'POST_DRAWER',  'tries':'INITIAL'}) 
        smach.StateMachine.add('POST_DRAWER',     Post_drawer(),  transitions = {'failed':'GRASP_DRAWER',    'succ': 'SCAN_TABLE2',  'tries':'END'}) 
        

        

      

    outcome = sm.execute()


    