import cv2
import numpy as np 

prey = [0.20532213, 0.5140056,  0.32135854]
predator = [0.60247678, 0.12115583, 0.09494324] # Red Picked 
wall = [0.8169659 , 0.75140956, 0.62361997] # Be Picked
ground = [0.65224764, 0.67061128, 0.68097509] # bluish gray

features = (np.array([predator,prey,ground,wall]) * 255).astype(np.uint8)
detection_thresholds = [20,30,20,30]

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

if __name__ == "__main__":
    img = cv2.imread("imgs/photos/record-4.png")
    edges,filled,mask,img = segment(img)
    cv2.imshow("edges",edges)
    cv2.imshow("filled",filled)
    cv2.imshow("mask",mask)
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()