import cv2
from cv2 import aruco
import numpy as np

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
print(dir(aruco))

size = 1000
offset = 100



def create_marker(dict,id):

    img = np.ones((size,size),dtype=np.uint8)*255
    img[offset:size-offset,offset:size-offset] = aruco.generateImageMarker(dict,id, size - 2*offset)
    img = cv2.putText(img, str(id), (int(size/2),int(size - offset/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2, cv2.LINE_AA)
    

    return img
    #cv2.imwrite('markers/marker_{}.png'.format(id),img)
    #cv2.imshow('frame',img)
    #cv2.waitKey(0)

def stack_imgs(imgs):
    
    img = np.concatenate(imgs,axis=0)

    return img





# second parameter is id number
ids = list(range(50))

sample = np.random.choice(ids, 25, replace=False)

for i in range(len(sample) // 2):

    img1 = create_marker(aruco_dict, i)
    img2 = create_marker(aruco_dict, i + 25 // 2)

    line = np.zeros((2, size), dtype=np.uint8)
    img = stack_imgs([img1,line, img2])

    cv2.imwrite('markers/marker_{}.png'.format(i), img)
    #cv2.imshow('frame', img)
    #cv2.waitKey(0)


