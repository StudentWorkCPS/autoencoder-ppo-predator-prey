import cv2 
import matplotlib.pyplot as plt





def detect_aruco(img):
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict,
        parameters=arucoParams)

    centers = []]

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

            # draw the bounding box of the ArUCo detection
            img = cv2.line(img, topLeft, topRight, (0, 255, 0), 2)
            img = cv2.line(img, topRight, bottomRight, (0, 255, 0), 2)
            img = cv2.line(img, bottomRight, bottomLeft, (0, 255, 0), 2)
            img = cv2.line(img, bottomLeft, topLeft, (0, 255, 0), 2)

            #direction
            #img = cv2.tri

            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            center = (markerID,cX,cY)
            centers.append(center)
            cv2.circle(img, (cX, cY), 4, (0, 0, 255), -1)

            # draw the ArUco marker ID on the image
            cv2.putText(img, str(markerID),
                (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
            print("[INFO] ArUco marker ID: {}".format(markerID))

    return img, centers

cap = cv2.VideoCapture('GH010128.MP4')
img = None

centers_history = {}

plt.ion()
steps = 0
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
    
        #break
    else:
        img = frame

    img,centers = detect_aruco(img)


    if centers is not None:

        for center in centers:
            if center[0] not in centers_history:
                centers_history[center[0]] = [(None,None)] * steps
            centers_history[center[0]].append((center[1],center[2]))    

    
    img_resized = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
    steps += 1
    
    #cv2.imshow('frame', img_resized)
    print(centers_history)

    plt.subplot(1,2,1)
    plt.imshow(img_resized)
    plt.subplot(1,2,2)
    plt.plot([x[0] for x in centers_history], [x[1] for x in centers_history])
    plt.draw()

    plt.pause(0.0001)
    plt.clf()

   
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
