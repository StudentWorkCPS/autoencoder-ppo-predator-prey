import cv2
import numpy as np
import os

DIM=(2704, 1520)
K=np.array([[1246.1215620841267, 0.0, 1345.5736937224221], [0.0, 1245.0562112685095, 769.197497945578], [0.0, 0.0, 1.0]])
D=np.array([[0.055202581328905556], [-0.0001264649695510569], [0.03248778211855575], [-0.028181626608413377]])

POS_MAP = {
    "2Cal1.mp4": {
        13: (0,-200+11.1,0),
        3: (200 - 11.9, 0,0),
        5: (0, 200 - 11.6,0),
        2: (-200 + 11.1, 0,0),   
    },
    "2Cal2.mp4": {
        2: (-200+12.4, -200+11,0),
        13: (200 - 11.2, -200 + 11.5,0),
        3: (200 - 11, 200-12.2,0),
        5: (-200 + 12, 200-12,0),
    },
    "2Cal3.mp4": {
        3: (0,0,0)
    }
}

P = None
if 'projection.npy' in os.listdir():
    P = np.load('projection.npy')

def project(pxl_points):
    pxl_points = np.array(pxl_points).astype(np.float64)
    if P is None:
        print('No projection matrix found')
        return None
    
    pxl_points = cv2.undistortPoints(pxl_points, K, D)
    pxl_points = np.squeeze(pxl_points)

    return np.dot(pxl_points, P)



def undistort(img):
    h,w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,D,(w,h),1,(w,h))
    dst = cv2.undistort(img, K, D, None, newcameramtx)
    # crop the image
    #x,y,w,h = roi
    #dst = dst[y:y+h, x:x+w]
    return dst

def detect_aruco(img,side='topRight'):

    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict,
        parameters=arucoParams)
    
    centers = []

    if len(corners) > 0:
        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4,2))

            
            
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))


            # position estimation of the marker
            
            
            # draw the bounding box of the ArUCo detection
            img = cv2.line(img, topLeft, topRight, (0, 255, 0), 2)
            img = cv2.line(img, topRight, bottomRight, (0, 255, 0), 2)
            img = cv2.line(img, bottomRight, bottomLeft, (0, 255, 0), 2)
            img = cv2.line(img, bottomLeft, topLeft, (0, 255, 0), 2)


            
            #direction
            #img = cv2.tri

            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2)
            cY = int((topLeft[1] + bottomRight[1]) / 2)

            switcher = {
                'topRight': topRight,
                'topLeft': topLeft,
                'bottomRight': bottomRight,
                'bottomLeft': bottomLeft,
                'center': (cX,cY)
            }
            

            pos = switcher[side]
            img = cv2.circle(img, pos, 5, (0, 0, 255), -1)
            angle = np.arctan2(topRight[1] - bottomRight[1],topRight[0] - bottomRight[0])
            center = (markerID,pos[0],pos[1],angle)
            centers.append(center)
            #cv2.circle(img, (cX, cY), 4, (0, 0, 255), -1)

            # draw the ArUco marker ID on the image
            cv2.putText(img, str(markerID),
                (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
            #print("[INFO] ArUco marker ID: {} POS {}".format(markerID,center))

    
    '''
    for (id1,id2) in plane:
        c1 = [c for c in centers if c[0] == id1][0]
        c2 = [c for c in centers if c[0] == id2][0]

        cv2.line(img, (c1[1],c1[2]), (c2[1],c2[2]), (0, 255, 0), 2)
    '''



    return img, centers

def get_pixel_to_real():

    files = POS_MAP.keys()

    real_points = []
    pxl_points = []

    for f in files:
        cap = cv2.VideoCapture('aruco-videos/' + f)
        ret = False

        while (cap.isOpened()):
            ret,img = cap.read()   
            #print(img) 
            if ret == True:
                break

        cap.release()
        
        img,centers = detect_aruco(img)

        for id,cx,cy in centers:

            if id in POS_MAP[f]:
                real_pt = POS_MAP[f][id]

                real_points.append([real_pt[0],real_pt[1],real_pt[2]])
                pxl_points.append([cx,cy])

    return np.array(pxl_points).astype(np.float64),np.array(real_points).astype(np.float64)





def begin_calibrate():
    pxl_points,real_points = get_pixel_to_real()

    pxl_points = cv2.undistortPoints(pxl_points, K, D)
    pxl_points = np.squeeze(pxl_points)

    projection = np.linalg.lstsq(
        pxl_points, real_points, rcond=-1)[0]
    
    np.save('projection.npy', projection)
    
    print("PROJECTION MATRIX:",projection)
    est = np.dot(pxl_points, projection)
    error = np.linalg.norm(real_points - est,axis=1)
    print("ERROR:",error.mean())
    print("Successfully calibrated camera")
    


    
if __name__ == "__main__":
    begin_calibrate()