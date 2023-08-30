import cv2 
import matplotlib.pyplot as plt
import numpy as np

print(cv2   .__version__)

# Distortion values 
DIM=(2704, 1520)
K=np.array([[1246.1215620841267, 0.0, 1345.5736937224221], [0.0, 1245.0562112685095, 769.197497945578], [0.0, 0.0, 1.0]])
D=np.array([[0.055202581328905556], [-0.0001264649695510569], [0.03248778211855575], [-0.028181626608413377]])

# GoPro 7 Black
c_real_height = 2.85

camera_pos = (-10,60,2.85)

c_to_wall_angle = 20 # degrees



c_pxl_width = 1520 # px
c_pxl_height = 2704 # px
# mm to m
# 

plane = np.array([(4,2),(2,1),(1,5),(5,4)])

sensor_width = 6170 # mm
sensor_height = 4550 # mm


sensor_d = np.sqrt(sensor_width**2 + sensor_height**2)

c_v_fov = 69.5 # degrees
c_h_fov = 118.2 # degrees

#
c_fx = sensor_d / (2 * np.tan(np.deg2rad(c_h_fov/2))) # m
c_fy = sensor_d / (2 * np.tan(np.deg2rad(c_v_fov/2))) # m

print('camera focal length x',c_fx)
print('camera focal length y',c_fy)


c_cx = c_pxl_width / 2 # px
c_cy = c_pxl_height / 2 # px

c_c = np.array([c_cx, c_cy])

c_matrix = np.array([[c_fx, 0, c_cx], [0, c_fy, c_cy], [0, 0, 1]]) 



marker_to_position = {
    3: [0.0,0.0,.0],
    4 : [-1000,1000,.0],  # left bottom
    2 : [-1000,-1000,.0], # left top
    1 : [1000,-1000,.0], # right top
    5 : [1000,1000,.0],  # right bottom
   # 16 : [0.2,0.1,.3],
}

    

def optimisation_problem(objs,pxls):

    import tensorflow as tf
    from tensorflow.python.ops.numpy_ops import np_config
    np_config.enable_numpy_behavior()

    objs = tf.constant(objs,dtype=tf.float64)
    wide_objs = tf.concat([objs, tf.constant([[1.] for i in range(len(objs))] ,dtype=tf.float64)], axis=1)
    pxls = tf.constant(pxls,dtype=tf.float64)
    wide_pxls = tf.concat([pxls, tf.constant([[1.,1.] for i in range(len(pxls))] ,dtype=tf.float64)], axis=1)
    wide_pxls = wide_pxls.astype(tf.float64)

    mat = tf.Variable(tf.eye(4,dtype=tf.float64))

    
    def dist(p1,p2):
        #print('p1',p1.shape)
        #print('p2',p2.shape)
        return tf.reduce_sum(tf.square(p1-p2))
    
    def project(p, mat):
        #print('p',p.T)
        p = tf.matmul(mat, p.T)
        #print('p',p.shape)
        p = tf.divide(p[:3], p[3])
        p = tf.transpose(p)
        return p
    
    
    def evaluate(mat,objs,wide_pxls):
        total = 0
        #print('mat',mat.shape)
        #print('wide_pxls',wide_pxls.shape)
        #tmp = wide_pxls[0,:]
        #print('tmp',tmp)
        #wide_pxls = np.zeros((len(objs),4))
        #wide_pxls[0,:] = tmp
        #wide_pxls[:,3] = 1
        #wide_pxls = tf.constant(wide_pxls,dtype=tf.float64)

        projected = project(wide_pxls,mat)

        print(objs,projected,wide_pxls)
        return dist(objs,projected)
        
    
    for i in range(3):
        with tf.GradientTape() as tape:
            est = evaluate(mat,objs,wide_pxls)
        grads = tape.gradient(est, mat)
        #print('grads',grads)
        mat.assign_sub(grads * 0.01)
        print(f'Epoch {i}: \t {est}')
        if i % 100 == 0:
            print(f'Epoch {i}: \t {est}')

    return mat


def monte_carlo(objs,pxls):
    mat = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    def project(p, mat):
        p = np.array(p)
        #print('p',p.shape)
        p = np.append(p, np.array([1,1]))
        p = p.reshape(-1,1)
        #print('p',p.shape)
        p = mat @ p
        p = p.reshape(-1)
        #print('p',p.shape)
        p = p[:3] / p[3]
        return p

    def dist(p1,p2):
        return np.sum(np.square(p1-p2))

    def evaluate(mat,objs,pxls):
        total = 0
        for i in range(len(objs)):
            projected = project(pxls[i],mat)
            total += dist(objs[i],projected)
        return total
    
    def perturb(mat,amount):
        mat2 = mat.copy()
        mat2[np.random.randint(0,4),np.random.randint(0,4)] += np.random.uniform(-amount,amount)

        return mat2
    
    def approximate(mat, amount, n=1):
        est = evaluate(mat,objs,pxls)

        for i in range(n):
            mat2 = perturb(mat,amount)
            est2 = evaluate(mat2,objs,pxls)
            if est2 < est:
                #print('est',est2)
                mat = mat2
                est = est2

        return mat, est

    for i in range(100):
        #print('i',i)
        mat,est = approximate(mat, 1)
        mat,est = approximate(mat, .1)
        print('est',est,'i',i)

    print('mat',mat)
    return mat

def linear_approximation_world(point ,edges,real_world):
    print('point',point.shape)
    print('edges',edges.shape)
    dist = edges - np.array(point)
    dist = dist.reshape(-1,point.shape[0])
    print('dist',dist.shape)
    print('dist',dist)
    dist = np.square(dist,dist)
    dist = np.sqrt(np.sum(dist,axis=1))
    
    total = np.sum(dist)
    print('total',total)
    print('dist',dist)

    null = np.argwhere(dist <= 0)

    if len(null) > 0:
        print('null',null)
        return real_world[null[0][0]]



    
    print('dist',dist)
    # normalize this vector to sum to 1
    mag = np.square(dist,dist)
    mag = np.sqrt(np.sum(mag))

    dist = np.array(dist) / mag
    dist = 1 - dist
    print('dist',dist)

    print('total',dist.shape,dist.sum(),real_world.shape)
    
    real_world = dist.reshape(1,-1) @ np.array(real_world)

    return np.sum(real_world,axis=0)



def pt_to_world(pt,R,t):
 
    pts = np.array(pt).reshape(-1,2)

    x = pts[:,0]
    y = pts[:,1]
    z = np.ones(len(x))

    camera_pt = np.array([x,y,z])
    print('camera_pt',camera_pt.shape)


    

    K_inv = np.linalg.inv(K)

    #camera_pt = np.dot(K,camera_pt)


    transformation_matrix = np.hstack((R, t))
    
    #transformation_matrix = np.dot(K_inv,transformation_matrix)
    # Add the last row to the transformation matrix [0, 0, 0, 1]
    transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1]))

    # Invert the Transformation Matrix to get the inverse transformation
    inverse_transformation_matrix = np.linalg.inv(transformation_matrix)

    #

    #inverse_transformation_matrix = np.dot(K_inv,inverse_transformation_matrix)

    # Convert the Camera Point to Homogeneous Coordinates
    homogeneous_camera_point = np.vstack((camera_pt, [1 for i in range(len(camera_pt[0]))]))

    # Perform the Transformation from Camera Space to World Space

    print('inverse_transformation_matrix',inverse_transformation_matrix)

    homogeneous_world_point = np.dot(inverse_transformation_matrix, homogeneous_camera_point)

    print('homogeneous_world_point',homogeneous_world_point)
    # Convert the result back to Inhomogeneous Coordinates (X_w, Y_w, Z_w)
    world_point = homogeneous_world_point[:3] / homogeneous_world_point[3]

    
    #world_point = np.dot(K_inv,world_point)

    #world_point = homogeneous_world_point[:3] / homogeneous_world_point[1]

    return world_point

def trie_area(trie):

    return 0.5 * (np.cross(trie[1]-trie[0], trie[2]-trie[0]))

def on_trie(uv,trie):

    area = trie_area(trie)

    a1 = trie_area([trie[0],trie[1],uv])
    a2 = trie_area([trie[1],trie[2],uv])
    a3 = trie_area([trie[2],trie[0],uv])

    if np.abs(a1 + a2 + a3 - area) < 0.00001:
        return True
    
    return False
    



def pxl_uv_to_world(u,v,pxls,real_world):
    # USE the uv map => 0,0 is top left
    tries = [(1,3,8),(1,3,7),(3,2,8),(3,2,6),(0,2,8),(0,2,5),(0,1,8),(0,1,4)]
    uv = np.array([u,v])
    for i,j,k in tries:
        trie = np.array([pxls[i],pxls[j],pxls[k]])
        
        if on_trie(uv,trie):
            print('trie',trie)
            real_pts = np.array([real_world[i],real_world[j],real_world[k]])
            return linear_approximation_world(uv,trie,real_pts)

    return None


def calibrate(points, pxl_points):
    # = cv2.calibrateCamera(points, pxl_points, DIM, K, D)
    #print(points.shape)
    #print(pxl_points.shape)
    #points = np.array([points])
    #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(points, pxl_points, DIM, K, D)

    #print("ret: ", ret)
    #print("mtx: ", mtx)
    #print("dist: ", dist)
    #print("rvecs: ", rvecs)
    #print("tvecs: ", tvecs)
    pass

def p_n_p(points,pxl_points):

    flag = cv2.SOLVEPNP_ITERATIVE # tried with SOLVEPNP_EPNP, same error.

    print(points.shape)
    print(pxl_points.shape)

    #pxl_points = camera_space(pxl_points)
    
    (success, rotation_vector, translation_vector) = cv2.solvePnP(points, pxl_points, K, D, flags=flag)
    print('rotation_vector',rotation_vector)
    R = cv2.Rodrigues(rotation_vector)[0]
    
    print(success)
    return np.matrix(R), np.matrix(translation_vector)

def undistort(img):    
    h,w = img.shape[:2]    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    #undistorted_img = cv2.undistort(img, K, D, None, K)

    return undistorted_img

def camera_space(pxl_pt):
    c_pt =  (pxl_pt - c_c) / np.array([c_fx, c_fy])
    #c_pt = np.append(c_pt, np.array([1 for i in range(len(c_pt))]).reshape(-1,1), axis=1)
    return c_pt


def detect_aruco(img):
    print('detect_aruco',dir(cv2.aruco))
    print('loser',dir(cv2))

    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict,
        parameters=arucoParams)
    
    tvecs = np.zeros((len(corners),3))
    rvecs = np.zeros((len(corners),3))

    ret = cv2.aruco.estimatePoseSingleMarkers(corners , 0.115, K, D, None, None)
    (rvec, tvec) = (ret[0][0, 0, :], ret[1][0, 0, :])

    for i in range(len(corners)):
        rvecs[i] = ret[0][i,0,:]
        tvecs[i] = ret[1][i,0,:]
        img = cv2.drawFrameAxes(img, K, D, rvecs[i], tvecs[i], 0.115)
    
    #print('rvec',l)


    

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


            img = cv2.circle(img, topRight, 5, (0, 0, 255), -1)
            #direction
            #img = cv2.tri

            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2)
            cY = int((topLeft[1] + bottomRight[1]) / 2)
            center = (markerID,topRight[0],topRight[1])
            centers.append(center)
            #cv2.circle(img, (cX, cY), 4, (0, 0, 255), -1)

            # draw the ArUco marker ID on the image
            cv2.putText(img, str(markerID),
                (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
            print("[INFO] ArUco marker ID: {} POS {}".format(markerID,center))

    
    '''
    for (id1,id2) in plane:
        c1 = [c for c in centers if c[0] == id1][0]
        c2 = [c for c in centers if c[0] == id2][0]

        cv2.line(img, (c1[1],c1[2]), (c2[1],c2[2]), (0, 255, 0), 2)
    '''



    return img, centers



pos_map = {
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

if __name__ == '__main__':

    files = pos_map.keys()
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

            if id in pos_map[f]:
                real_pt = pos_map[f][id]

                real_points.append([real_pt[0],real_pt[1],real_pt[2]])
                pxl_points.append([cx,cy])

    print('real_points',real_points)
    print('pxl_points',pxl_points)

    
    










    real_points = np.array(real_points).astype(np.float64)
    pxl_points = np.array(pxl_points).astype(np.float32)

    de_pxl_points = cv2.undistortPoints(pxl_points, K, D)
    de_pxl_points = de_pxl_points.reshape(-1,2)
    print('de_pxl_points',de_pxl_points)
    de_pxl_points = np.concatenate([de_pxl_points,np.zeros((len(de_pxl_points),1))],axis=1)


    #print('de_pxl_points',de_pxl_points)
    # PNO
    
    #mat = optimisation_problem(real_points,de_pxl_points)
    #print('mat',mat)
    #print('mat',mat.shape)

    #plt.subplot(1,2,1)
    #     plt.scatter(pxl_points[:,0],pxl_points[:,1])
    #ax = plt.subplot(1,2,2)
    #3d 
    #pxl_points = np.concatenate((pxl_points,np.ones((len(pxl_points),1))),axis=1)
    #import tensorflow as tf
    #wide_pxls = tf.concat([pxl_points, tf.constant([[1.,1.] for i in range(len(pxl_points))] ,dtype=tf.float64)], axis=1)
    
    #print('pxl_points',pxl_points.shape)
    #estimated_pts = mat @ wide_pxls.T
    
    #estimated_pts = estimated_pts[:3] / estimated_pts[3]
    #estimated_pts = estimated_pts.T
    #estimated_pts = estimated_pts.numpy()
    #estimated_pts[:,2] = 0
    #norm_est = ((estimated_pts - estimated_pts.min()) / (estimated_pts.max() - estimated_pts.min()) - 0.5 ) * 200
    #norm_est[:,2] = 0
    #print('norm_est',norm_est)
    #print('estimated_pts',estimated_pts)
    print('real_points',real_points)
    print('de_pxl_points',de_pxl_points)
    
    ax = plt.axes(projection='3d')
    #ax.scatter3D(real_points[:,0], real_points[:,1], real_points[:,2],label='real')
    zeros = np.zeros((len(real_points),3))
    print('zeros',zeros.shape)
    de_pxl_points = de_pxl_points.reshape(-1,2)
    print(de_pxl_points.shape)
    #ax.scatter(de_pxl_points[:,0], de_pxl_points[:,1],label='pxl')
    real_points = real_points / 200
    
    R,tvec = p_n_p(real_points,de_pxl_points)

    print('R',R)
    print('tvec',tvec)

    P = np.hstack((R,tvec))
    print('P',P)
    P = np.dot(K,P)
    P = np.vstack((P, [0, 0, 1, 0]))
    P_inv = np.linalg.inv(P)
    est = np.zeros((len(real_points),3))
    for i in range(len(real_points)):
        u = de_pxl_points[i,0] + np.random.uniform(-0.01,0.01)
        v = de_pxl_points[i,1] + np.random.uniform(-0.01,0.01)

        est[i] = pxl_uv_to_world(u,v,de_pxl_points,real_points)
        
    est = np.array([pxl_uv_to_world(0,0,de_pxl_points,real_points)])

    print('real',real_points)
    print('est',est)
    ax.scatter3D(est[:,0], est[:,1], est[:,2],label='est')
    de_pxl_points = np.concatenate([de_pxl_points,np.zeros((len(de_pxl_points),1))],axis=1)
    ax.scatter3D(real_points[:,0], real_points[:,1], real_points[:,2],label='real')
    ax.scatter3D(de_pxl_points[:,0], de_pxl_points[:,1], de_pxl_points[:,2],label='pxl')
    ax.legend()
    print(de_pxl_points.shape)
    
    for i in range(len(real_points)):
        ax.text(real_points[i,0], real_points[i,1], real_points[i,2], str(i))
        #ax.text(est[i,0], est[i,1], est[i,2], str(i))
        ax.text(de_pxl_points[i,0], de_pxl_points[i,1], de_pxl_points[i,2], str(i))
        #print(de_pxl_points)
        #ax.text(de_pxl_points[i,0], de_pxl_points[i,1], str(i), str(i))
        #continue
        #ax.text(de_pxl_points[i,0], de_pxl_points[i,1],np.zeros((len(de_pxl_points),)), str(i))

    #img =cv2.drawFrameAxes(img, K, D, R, tvec, 0.1)
    img = undistort(img)
    border = 100
    img = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=[0,0,0])
    cv2.imshow('img', img)

    plt.show()



    
    
    



    #img = undistort(img)

    #undistored_points = []
    #for id,cx,cy in centers:
    #    undistored_points.append((cx,cy))

    #undistored_points = np.array(undistored_points).astype(np.float32)
    #print('undistored_points',len(undistored_points))
    #undistored_points = cv2.undistortPoints(undistored_points, K, D)


    
    exit(0)
    

    real_points = []
    pxl_points = []

    undistored_points = np.array([[cx,cy] for (id,cx,cy) in centers]).astype(np.float32)
    print()
    undistored_points = cv2.undistortPoints(undistored_points, K, D) #* np.array([K[0,0], K[1,1]]) +  np.array([K[0,2], K[1,2]])
        

    for i,(id,cx,cy) in enumerate(centers):
        if id in marker_to_position:
            (x,y) = undistored_points[i].reshape(-1)
            print(f'ID: {id} CAM: {(cx,cy)} POS: {marker_to_position[id]}')
            real_points.append(marker_to_position[id])
            pxl_points.append([x,y])

    real_points = np.array(real_points).astype(np.float64)
    pxl_points = np.array(pxl_points).astype(np.float32)
    print('real_points',len(real_points))
    #try: 
    try:
        #calibrate(real_points,pxl_points)
        R,tvec = p_n_p(real_points,pxl_points)
        print('T', tvec)


        #extrinsic_matrix = np.hstack((R,tvec))


        #P = np.dot(c_matrix,extrinsic_matrix)
        #print('P',P.shape)
        #P_inv = np.linalg.inv(P)


        #print('PNP:',extrinic_matrix)
        wrl_x = []
        wrl_y = []
        wrl_z = []
        ids = []

        ax = plt.axes(projection='3d')

        R_inv = np.linalg.inv(K * R)  
        
        #better_c_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, DIM, np.eye(3), balance=0.0)

        #tvec = tvec.reshape(-1)

        
        #undistored_points = cv2.convertPointsToHomogeneous(undistored_points)

        real_world_markers = pt_to_world(pxl_points,R,tvec)
        print('real_world_markers',real_world_markers.shape)
        
        for i,(id,cx,cy) in enumerate(centers):

            print(f'ID: {id} CAM: {(cx,cy)}')
            

            
            # to world points = (R * pxl_points + t) * scale
            undistored_p = undistored_points[i].reshape(-1)
            #print('undistored_p',undistored_p)
            #undistored_p = camera_space(undistored_p)
            (x,y) = undistored_p
            wrld_points = pt_to_world((x,y),R,tvec)
        
            #wrld_points = linear_approximation_world(wrld_points,real_world_markers,real_points)
            #print('wrld_points',wrld_points)

            

            #print(wrld_points)
            #print('undistored_p',undistored_p)
            #c = np.array([[undistored_p[0]],[undistored_p[1]],[1]])
            #c = np.array([[un],[cy],[1]])
            #c = camera_space(c)
            #print('c',c.shape)
            #print('R_inv',R_inv.shape)
            #print('tvec',tvec.shape)
            #print('better_c_matrix',better_c_matrix)


            #wrld_points = (R_inv * c) -  tvec
            #wrld_points = # pt_to_world((x,y),R,tvec)
            #wrld_points[2] = 0

            wrl_x.append(wrld_points[0])
            wrl_y.append(wrld_points[1])
            wrl_z.append(wrld_points[2])
            ids.append(id)


            print(f'ID {id} wrld_points {wrld_points}')

        #ids.append(0)

        #camera_pos = np.array([0,0,0]).reshape(-1,1)
        #camera_pos = R_inv * np.array([[0],[0],[1]] - tvec)

        #wrl_x.append(camera_pos[0])
        #wrl_y.append(camera_pos[1])
        #wrl_z.append(camera_pos[2])


        ax.scatter3D(wrl_x, wrl_y, wrl_z, c=ids)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        '''        for (id1,id2) in plane:
            #i1 = [i for i,c in enumerate(centers) if c[0] == id1][0]
            #i2 = [i for i,c in enumerate(centers) if c[0] == id2][0]

            print('i1',i1)
            print('i2',i2)
            
            s = np.array([wrl_x[i1], wrl_y[i1], wrl_z[i1]]).reshape(-1,3)
            e = np.array([wrl_x[i2], wrl_y[i2], wrl_z[i2]]).reshape(-1,3)
             #(pt_to_world((wrld_points[i1][0],wrld_points[i1][1]),R,tvec).reshape(-1,3))
            #e = #(pt_to_world((wrld_points[i2][0],wrld_points[i2][1]),R,tvec).reshape(-1,3))

            real_s = np.array([marker_to_position[id1],marker_to_position[id2]]).reshape(-1,3) 
            real_e = np.array([marker_to_position[id2],marker_to_position[id1]]).reshape(-1,3)
            #ax.
            print('s',s.shape)
            print('e',e.shape)
            #print(dir(ax))
            
            #ax.plot3D([real_s[0,0],real_e[0,0]],[real_s[0,1],real_e[0,1]],[real_s[0,2],real_e[0,2]],'blue')

            ax.plot3D([s[0,0],e[0,0]],[s[0,1],e[0,1]],[s[0,2],e[0,2]],'red')
        '''


        #ax.quiver()
    
        print(ids)
        for i in range(len(ids)):
            print(ids[i], wrl_x[i], wrl_y[i], wrl_z[i])
        

            #ax.text(wrl_x[i], wrl_y[i], wrl_z[i], ids[i], zdir='z')

        #ax.texts(ids, wrl_x, wrl_y, wrl_z, zdir='z')
    
        #plt.draw()
        #plt.pause(0.0001)
    #plt.clf()
    except Exception as e:
        raise e

    #img = undistort(img)

    resized = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

    

    cv2.imshow('img', resized)
    plt.show()
    #cv2.waitKey(0)
