def lineup_table():
    
    cv2_img=rgbd.get_image()
    img=cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    img=cv2.Canny(img,80,200)
    lines = cv2.HoughLines(img,1,np.pi/180,200)
    first=True
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            size=12
            if first:
                size=52
                first=False

            img=cv2.line(img,(x1,y1),(x2,y2),(255,255,255),size)

    #
    plt.imshow(img)
    wb=whole_body.get_current_joint_values()
    wb[2]+=-(lines[0,0,1]-.5*np.pi)
    succ=whole_body.go(wb)
    return succ
