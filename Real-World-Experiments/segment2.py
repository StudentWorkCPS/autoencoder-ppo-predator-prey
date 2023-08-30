import segment as seg
import numpy as np
import cv2
import matplotlib.pyplot as plt


thresholds = [10,10,10]

def in_bound(x,min,max):
    return x >= min and x <= max


class Feature:
    def __init__(self,min,max,color,y):
        self.min = min
        self.max = max
        self.color = color
        self.minY = y[0]
        self.maxY = y[1]

    def in_range(self,img):
        return cv2.inRange(img, self.min, self.max) >= 1
    
    def in_bounds(self,y):
        return in_bound(y,self.minY,self.maxY)
    
robot_min = np.array([0,0,0]).astype(np.uint8)
robot_max = np.array([45,45,45]).astype(np.uint8)
robot_feature = Feature(robot_min,robot_max,[255,255,255],(35,64))

prey_min = np.array([14,55,26]).astype(np.uint8)
prey_max = np.array([60,99,60]).astype(np.uint8)
prey_feature = Feature(prey_min,prey_max,[0,255,0],(31,64))

predator_min = np.array([114,3,3]).astype(np.uint8)
predator_max = np.array([150,23,20]).astype(np.uint8)
predator_feature = Feature(predator_min,predator_max,[255,0,0],(30,64))

ground_min = np.array([110,110,131]).astype(np.uint8)
ground_max = np.array([142,148,158]).astype(np.uint8)
ground_feature = Feature(ground_min,ground_max,[0,100,120],(50,64))


predator_hsv_min = np.array([0,175,100]).astype(np.uint8)
predator_hsv_max = np.array([20,240,175]).astype(np.uint8)
predator_hsv_feature = Feature(predator_hsv_min,predator_hsv_max,[255,0,0],(30,64))

prey_hsv_min = np.array([40,100,50]).astype(np.uint8)
prey_hsv_max = np.array([80,200,120]).astype(np.uint8)
prey_hsv_feature = Feature(prey_hsv_min,prey_hsv_max,[0,255,0],(30,64))



wall_min = np.array([146,137,117]).astype(np.uint8)
wall_max = np.array([197,189,167]).astype(np.uint8)

wall_feature = Feature(wall_min,wall_max,[200,200,100],(20,45))
normal_features = [ground_feature,wall_feature]

rest_feature = Feature(np.array([0,0,0]).astype(np.uint8),np.array([0,0,0]).astype(np.uint8),[0,0,0],(0,64))
total_features = [ground_feature,wall_feature,rest_feature,prey_feature,predator_feature]

thresholds = (10,10,10)

def flood_fill2(img,features):
    mask = np.zeros((64+2,64+2),np.uint8)
    range_img = np.zeros((64,64,3),np.uint8)
    for i in range(0,len(features)):
        feature = features[i]

        low = feature.min
        high = feature.max
        in_range = cv2.inRange(img, low, high) >= 1
        range_img[in_range] = feature.color
        arr = np.argwhere(in_range)
        
        if (len(arr) == 0):
            continue

        for j in range(0,len(arr)):
            if arr[j,0] < feature.minY or arr[j,0] > feature.maxY:
                continue
            #print(feature_color[i])
            rgb = (int(feature.color[0]),int(feature.color[1]),int(feature.color[2]))
            #print(rgb)
            _,img,mask,_ = cv2.floodFill(img, mask, (arr[j,1],arr[j,0]), rgb,loDiff=thresholds,upDiff=thresholds)

    #img = img * (mask[1:-1,1:-1] == 1)[:,:,np.newaxis]

    return img, mask,range_img




def blob_detection(img,feature):
    
    in_range = cv2.inRange(img, feature.min, feature.max) >= 1
    arr = np.argwhere(in_range)
    blobs = []
    for j in range(0,len(arr)):
        for i in range(0,len(blobs)):
            dist_min = np.linalg.norm(np.array(arr[j]) - np.array(blobs[i]['points']),axis=1).argmin()
            min_point = blobs[i]['points'][dist_min]
            dist = min_point - arr[j]

            #print(arr[j],blobs[i]['points'])
            #print(dist_min)
            if in_bound(dist[0],-3,3) and in_bound(dist[1],-3,3):
                blobs[i]['points'].append(arr[j])
                blobs[i]['center'] = np.mean(blobs[i]['points'],axis=0)
                break
        else:
            blobs.append({
                'points': [arr[j]],
                'center': arr[j],
            })
    #print("BEFOR FILTERING:",blobs)
    blobs = list(filter(lambda x: len(x['points']) > 3 and feature.in_bounds(x['center'][0]),blobs))
    #print('BLOBS:',blobs)

    return blobs,in_range

def segment2(img):
    edges = seg.compute_edges(img/255.0)
    

    flood_filled = cv2.resize(img, (64,64))

    flood_filled, mask1,range_img1 = flood_fill2(flood_filled,[prey_feature,predator_feature])

    mask1 = (mask1[1:-1,1:-1] == 1)[:,:,np.newaxis]

    remember = flood_filled * mask1 
    

    flood_filled = flood_filled * (edges == 0)[:,:,np.newaxis]

    flood_filled, mask2,range_img = flood_fill2(flood_filled,normal_features)

    mask2 = (mask2[1:-1,1:-1] == 1)[:,:,np.newaxis]
    mask = np.clip(mask1 + mask2,0,1)
    flood_filled = flood_filled * (mask == 1)
    flood_filled = cv2.dilate(flood_filled, np.ones((3,3),np.uint8), iterations=1)

    mask1 = np.concatenate([mask1,mask1,mask1],axis=2)
    flood_filled[mask1] = remember[mask1]

    flood_filled = cv2.dilate(flood_filled, np.ones((3,3),np.uint8), iterations=1)
    return flood_filled, mask ,edges,range_img + range_img1

def blob_fill(img,feature):

    blobs,in_range = blob_detection(img,feature)
    mask = np.zeros((64,64,1),np.uint8)

    for i in range(0,len(blobs)):
        min = np.min(blobs[i]['points'],axis=0)
        max = np.max(blobs[i]['points'],axis=0)

        img = cv2.rectangle(img,(min[1],min[0]),(max[1],max[0]),feature.color,-1)
        mask[min[0]:max[0],min[1]:max[1],0] = 1
        #print('BLOB:',min,max,feature.color)



    return img,in_range,mask

def one_hot_encode(img,features):

    img = img.reshape((-1,3))
    img = np.array(np.argmin([np.sum(np.abs(img - feature.color),axis=1) for feature in features],axis=0))
    img = np.eye(len(features))[img]
    return img.reshape((64,64,-1))


def segment3(img,DEBUG=False):
    edges = seg.compute_edges(img/255.0)
    blurred = cv2.GaussianBlur(img,(5,5),0)

    canny = cv2.Canny((blurred).astype(np.uint8),30,50)
    canny = cv2.dilate(canny, np.ones((3,3),np.uint8), iterations=1) >= 1
    canny = cv2.resize(canny.astype(np.uint8), (64,64))

    edges = np.clip(edges + canny,0,1)

    flood_filled = cv2.resize(img, (64,64))

    flood_filled_hsv = cv2.cvtColor(flood_filled,cv2.COLOR_RGB2HSV)
    
    remember1,in_range1, mask1 = blob_fill(flood_filled_hsv,prey_hsv_feature)
    remember2,in_range2, mask3 = blob_fill(flood_filled_hsv,predator_hsv_feature)

    remember = remember1 * mask1 +  remember2 * mask3


    in_range2 = in_range2.reshape(64,64,1) * predator_feature.color
    in_range1 = (in_range1.reshape(64,64,1) * prey_feature.color)
    in_range1 = (in_range2 + in_range1).astype(np.uint8)

    mask1 = np.clip(mask3 + mask1,0,1)
    mask1 = (mask1 == 1)[:,:] # [1:-1,1:-1]
    remember = remember * mask1 

    flood_filled = flood_filled * (edges == 0)[:,:,np.newaxis]

    flood_filled, mask2,range_img = flood_fill2(flood_filled,normal_features)

    mask2 = (mask2[1:-1,1:-1] == 1)[:,:,np.newaxis]
    mask = np.clip(mask1 + mask2,0,1)
    #print(mask.shape)
    #print(flood_filled.shape)
    flood_filled = flood_filled * (mask == 1)
    #print(flood_filled.shape)
    flood_filled = cv2.dilate(flood_filled, np.ones((3,3),np.uint8), iterations=1)

    mask1 = np.concatenate([mask1,mask1,mask1],axis=2)
    flood_filled[mask1] = remember[mask1]


    if DEBUG:
        np.zeros((64,64,3),np.uint8)
        for feature in normal_features + [prey_feature,predator_feature]:
            
            cv2.line(flood_filled,(0,feature.minY - 1),(64,feature.minY - 1),feature.color,1)
            cv2.line(flood_filled,(0,feature.maxY - 1),(64,feature.maxY - 1),feature.color,1)

    segmented = one_hot_encode(flood_filled,total_features)

    #flood_filled = cv2.dilate(flood_filled, np.ones((3,3),np.uint8), iterations=1)
    return segmented,flood_filled, mask ,edges,range_img + in_range1,remember


if __name__ == "__main__":
    img = cv2.imread("output.png")
    to_look_into = img.copy()
    img = cv2.resize(img, (64,64),interpolation=cv2.INTER_NEAREST)
    img,mask,edges,range_img = segment2(img)
    #plt.figure()
    for i,im in enumerate([img,mask[1:-1,1:-1],edges,range_img]):
        plt.subplot(1,4,i+1)
        plt.imshow(im)
        print(im.shape)