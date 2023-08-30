import cv2
import numpy as np 

prey = [0.20532213, 0.5140056,  0.32135854]
predator = [0.60247678, 0.12115583, 0.09494324] # Red Picked 
wall = [0.8169659 , 0.75140956, 0.62361997] # Be Picked
ground = [0.65224764, 0.67061128, 0.68097509] # bluish gray
robotOrElse = [0.0,0.0,0.0]

features = (np.array([predator,prey,ground,wall,robotOrElse]) * 255).astype(np.uint8)
detection_thresholds = [20,30,20,30,30]

def compute_edges(_img):
    #plt.imshow(_img)
    thresh = 0.3
    contrasted_left = cv2.filter2D(_img,-1,np.array([[+1,0,-1],[+2,0,-2],[+1,0,-1]])) > thresh
    contrasted_top = cv2.filter2D(_img,-1,np.array([[+1,+2,+1],[0,0,0],[-1,-2,-1]])) > thresh
    contrasted_bottom = cv2.filter2D(_img,-1,np.array([[-1,-2,-1],[0,0,0],[+1,+2,+1]])) > thresh
    contrasted_right = cv2.filter2D(_img,-1,np.array([[-1,0,+1],[-2,0,+2],[-1,0,+1]])) > thresh

    
    total_edges = contrasted_top + contrasted_bottom + contrasted_left + contrasted_right
    #print(total_edges.shape)
    total_edges = total_edges.any(axis=2).astype(float)
    total_edges = np.ones_like(total_edges) * total_edges

    
    total_edges = cv2.dilate(total_edges, np.ones((3,3),np.uint8), iterations=1)
    total_edges = cv2.resize(total_edges, (64,64),interpolation=cv2.INTER_NEAREST)

    return total_edges

def flood_fill(_img,features,thresholds):
    img = _img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for idx in range(len(features)):
                feature = features[idx]
                if (np.allclose(img[i][j],feature,atol=thresholds[idx]) and (img[i][j] != feature).all()) :
                    #print(feature)
                    rgb = (int(feature[0]),int(feature[1]),int(feature[2]))
                    img = cv2.floodFill(img, None, (j,i),rgb,loDiff=(10,10,10),upDiff=(10,10,10))[1]
    return img 



def generate_mask(img,features):
    result = np.zeros((64,64,1))
    for feature in features:
        result += (img == feature).all(axis=2).astype(np.uint8).reshape(64,64,1)

    
    return result

def segment(_img):

    img = np.array(_img).copy()

    edges = compute_edges(img/255.0)
    #plt.imshow(all_edges,cmap='gray')
    img = np.array(cv2.resize(img, (64,64),interpolation=cv2.INTER_NEAREST))

    # Add the edges to the image
    img = img * (edges == 0).reshape(64,64,1)
    
    filled = flood_fill(img,features ,detection_thresholds)
    
    mask = generate_mask(filled,features)
    #plt.imshow(mask,cmap='gray')
    #plt.imshow(mask)
    img = filled * (mask == 1)
    img = cv2.dilate(img, np.ones((3,3),np.uint8), iterations=1)
    
    return edges,filled,mask,img

def one_hot(_img,features):
    img = _img.reshape(-1,3)
    img = np.array(np.argmin([np.sum((img - feature)**2,axis=1) for feature in features],axis=0))
    img = np.eye(len(features))[img]


    img = img.reshape(64,64,-1)
    
    return img

def one_hot_segment(_img):

    img = cv2.resize(_img, (64,64),interpolation=cv2.INTER_NEAREST)

    one = one_hot(img,features)

    print(one.shape)
    print(features.shape)
    print((one @ features).shape)

    return None,None,None,(one @ features).astype(np.uint8)
    
#features = seg.features

def segment2(img):
    edges = compute_edges(img/255.0)
    flood_filled = cv2.resize(img, (64,64))

    flood_filled = flood_filled * (edges == 0)[:,:,np.newaxis]

    flood_filled, mask,range_img = flood_fill2(flood_filled)

    flood_filled = cv2.dilate(flood_filled, np.ones((3,3),np.uint8), iterations=1)

    return flood_filled, mask ,edges,range_img

thresholds = [10,10,10]

def flood_fill2(img):
    mask = np.zeros((64+2,64+2),np.uint8)
    range_img = np.zeros((64,64,3),np.uint8)
    for i in range(0,len(features)):
        low = (features[i] - thresholds)
        high = (features[i] + thresholds)
        in_range = cv2.inRange(img, low, high) >= 1
        range_img[in_range] = features[i]
        arr = np.argwhere(in_range)
        
        if (len(arr) == 0):
            continue

        for j in range(0,len(arr)):
            rgb = (int(features[i,0]),int(features[i,1]),int(features[i,2]))
            _,img,mask,_ = cv2.floodFill(img, mask, (arr[j,1],arr[j,0]), rgb,loDiff=thresholds,upDiff=thresholds)

    img = img * (mask[1:-1,1:-1] == 1)[:,:,np.newaxis]

    return img, mask,range_img

if __name__ == "__main__":
    img = cv2.imread("imgs/photos/record-4.png")
    import time
    start_time = time.time()
    edges,filled,mask,img_r = segment(img)
    result_time = time.time() - start_time
    print("Time Taken: ",result_time, 's', 'FPS: ', 1/result_time)
    img = np.array(cv2.resize(img, (64,64),interpolation=cv2.INTER_NEAREST))/255.0

    result_img = np.concatenate((img,img_r),axis=1)
    print(result_img.shape)
    cv2.imshow("Segmented",result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()