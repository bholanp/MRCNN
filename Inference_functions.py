
import cv2
import numpy as np
import pandas as pd


####### MANUALLY ADDED #################################
def mask2polygon(image,masks):
    _,_,numasks=masks.shape
    polygons=[]
    for i in range(numasks):
        mask_reformed=masks[:,:,i].astype("uint8")
        if np.sum(mask_reformed==1): # if mask is not empty - i.e. filled with zeros only, then proceed ..  
            contours, _ = cv2.findContours(image=mask_reformed, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE) # or you can also method=cv2.CHAIN_APPROX_NONE
            contours = tuple(contours)
            polygons.append(contours)
    return polygons
def centerofbox (boxs):
    centers=[]
    for cor in boxs:
        x,y = ((cor[1]+cor[3])/2),((cor[0]+cor[2])/2)
        a,b = int(x),int(y)
        centers.append([a,b])
    centers= np.asarray(centers, dtype=(int))
    return centers
def centerofmask (points):
    centers = []
    minx = []
    maxx= []
    extracted_points = []
    for i in points:
        p= i[np.argmin(i[:,0])]
        q= i[np.argmax(i[:,0])]
        minx.append(p)
        maxx.append(q)
        M = cv2.moments(i)
        if M['m00']!= 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centers.append([cx,cy])
    centers= np.asarray(centers, dtype=(int))
    minx= np.asarray(minx, dtype=(int))
    maxx= np.asarray(maxx, dtype=(int))
    return centers, minx, maxx
def pointsbetweenminxandmaxx(points,minx,maxx):
    listofcoor = []
    for i in range(len(points)):
        coor= []
        for j in range(len(points[i])):
            if((points[i][j][0]>= minx[i][0]and(points[i][j][1]<= minx[i][1]))or
               (points[i][j][0]<= maxx[i][0]and(points[i][j][1]<= maxx[i][1]))):
                coor.append(points[i][j])
        coor= np.asarray(coor, dtype=(int))
        listofcoor.append(coor)
    return listofcoor
def convertpolygonintopoints(polygons):
    points=[]
    for i in range(len(polygons)):
        pts=[]
        for j in range (len(polygons[i][0])):
            pts.append([int(polygons[i][0][j][0][0]),int(polygons[i][0][j][0][1])])
        pts= np.array(pts)
        points.append(pts)
    return points
'''def drawimage(image, points, boxes, center, maskcenter,minmax_x_point, miny, maxy, excoor, xeroslope):
    for i in range(len(points)):
        #cv2.drawContours(image=image, contours=[points[i]], contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA) # check the contours
        #image= cv2.circle(image, (center[i]), radius=3, color=(0, 0, 255), thickness=-1)
        image= cv2.circle(image, (maskcenter[i]), radius=3, color=(0, 0, 0), thickness=-1)
        #y1,x1,y2,x2 = boxes[i]
        #cv2.rectangle(image, (x1-10,y1-10), (x2+10,y2+10), (255,0,0),2)
        #cv2.drawContours(image=image, contours=[minmax_x_point[i]],contourIdx=-1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        #image = cv2.line(image, (maxy[i]), (maskcenter[i]), (0, 255, 0), 2)
        image = cv2.line(image, (xeroslope[i]), (excoor[i]), (0, 0, 255), 2)
        #image= cv2.circle(image, (miny[i]), radius=3, color=(0, 0, 255), thickness=-1)
        #image= cv2.circle(image, (maxy[i]), radius=3, color=(0, 0, 0), thickness=-1)
        image= cv2.circle(image, (excoor[i]), radius=3, color=(255, 255, 255), thickness=-1)
        image = cv2.putText(image,str(i),(excoor[i]),fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color = (255, 0, 0), thickness = 1)
        
    return image
'''
def drawimage(image, center, targetpoint, excoor, boxes, downpoint):
    for i in range(len(targetpoint)):
       
        image= cv2.circle(image, (center[i]), radius=3, color=(0, 0, 0), thickness=1)
        image = cv2.line(image, (targetpoint[i]), (excoor[i]), (255, 0, 0), 3)
        image= cv2.circle(image, (excoor[i]), radius=3, color=(255, 255, 255), thickness=2)
        image = cv2.putText(image, str(i),(excoor[i]),fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color = (0, 0, 255), thickness = 1)
        y1,x1,y2,x2 = boxes[i]
        cv2.rectangle(image, (x1,y1), (x2,y2), (255,255,255),2)
        for point1, point2 in zip(downpoint[i], downpoint[i][1:]): 
            cv2.line(image, point1, point2, [0, 255, 0], 2) 
    return image
def pointsbetweenminxandmaxxdown(points,minx,maxx):
    listofcoor = []
    for i in range(len(points)):
        coor= []
        for j in range(len(points[i])):
            if((points[i][j][0]>= minx[i][0]and(points[i][j][1]>= minx[i][1]))or
               (points[i][j][0]<= maxx[i][0]and(points[i][j][1]>= maxx[i][1]))):
                coor.append(points[i][j])
        coor= np.asarray(coor, dtype=(int))
        listofcoor.append(coor)
    return listofcoor
def minmaxy (points):
    miny = []
    maxy= []
    for i in points:
        p= i[np.argmin(i[:,1])]
        q= i[np.argmax(i[:,1])]
        for i in range(len(points)):
            coor= []
            for j in range(len(points[i])):
                if (points[i][j][1] == q[1]):
                    coor.append(points[i][j][1])
        miny.append(p)
        maxy.append(q)
    miny= np.asarray(miny, dtype=(int))
    maxy= np.asarray(maxy, dtype=(int))  
    return  miny, maxy
def pointscalculation (img, center, point, boxes):
    height = img.shape[0]
    width = img.shape[1]
    coor = []
    for i in range(len(point)):
        temp = []
        p1 = point[i]
        p2 = center[i]
        theta = np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
        y1,x1,y2,x2 = boxes[i]
        
        for i in range(50,500,5):    
            endpt_x = int(p1[0] - i*np.cos(theta))
            endpt_y = int(p1[1] - i*np.sin(theta))
            if (endpt_y <= (y1-10)):
                break
        if endpt_x <0: endpt_x=0
        if endpt_y <0: endpt_y=0
        if endpt_x >width: endpt_x=width
        if endpt_y >height: endpt_y=height
        temp = [endpt_x, endpt_y]
        temp = np.asarray(temp, dtype=(int))
        coor.append(temp)
    coor= np.asarray(coor, dtype=(int))
    return coor
def calculateslope(minmaxdown):
    slope = []
    points1 = []
    points2 = []
    for i in range (len(minmaxdown)):
        tempcoor = []
        temppoints1 = []
        temppoints2 = []
        for j in range (0,len (minmaxdown[i]),10):
            if ((j+10)>= len(minmaxdown[i])):
                break
            x1,y1 = minmaxdown[i][j]
            x2,y2 = minmaxdown[i][j+10]
            if x2 == x1:
                sslope =1000
            else:
                sslope = (y2-y1)/(x2-x1)
            temppoints1.append([x1,y1])
            temppoints2.append([x2,y2])
            tempcoor.append(sslope)
        temppoints1= np.asarray(temppoints1, dtype=(int))
        temppoints2= np.asarray(temppoints2, dtype=(int))
        points1.append(temppoints1)
        points2.append(temppoints1)
        slope.append(tempcoor)  
    coor = []
    for m in range(len(slope)):
        absslope = [abs(ele) for ele in slope[m]]
        minvalue = min(absslope)
        for n in reversed(range(len(slope[m]))):
            if ((slope[m][n]==minvalue) or (slope[m][n]==(-(minvalue)))):
               coor.append(points2[m][n])
               break
    coor = np.asarray(coor, dtype = (int))
    return coor
   
def findvortex (minmaxdown):
    vortex = []
    for i in range(len(minmaxdown)):
        data = pd.DataFrame(minmaxdown[i], columns = ["x", "y"])
        #polymod = np.poly1d(np.polyfit(data['x'], data['y'], 2))
    
        
        #NEW CODE FROM HERE
        p = np.polyfit(data['x'], data['y'], 2)
        xfit=np.linspace(min(data['x']),max(data['x']), 1000)
        yfit=np.polyval(p, xfit)
        
        polymod = np.poly1d(np.polyfit(xfit, yfit, 2))
        
        a,b,c = polymod.coef
        
        #Calculating X,Y as vortex
        x = ((-(b))/(2*a))
        y = polymod(x)
        
        #Calculating X,Y as focus
        #x = ((-(b))/(2*a))
        #y = (((1-(b**2))/(4*a))+c)
        
        coor = [x,y]
        coor = np.asarray(coor, dtype = "int")
        vortex.append(coor)
    vortex = np.asarray(vortex, dtype = "int")
    return vortex
        
def newdrawimage(image, center, points):
    for i in range(len(center)):
        
        cv2.drawContours(image=image, contours=[points[i]], contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    
        image= cv2.circle(image, (center[i]), radius=8, color=(255, 255, 255), thickness=8)
        image= cv2.circle(image, (center[i]), radius=3, color=(0, 0, 255), thickness=5)
        
    return image
    
