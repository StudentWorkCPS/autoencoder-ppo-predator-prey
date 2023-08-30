import segment2 as seg
import glob as glob
import domain_randomization as dr
import cv2
import numpy as np
import os

pngs = glob.glob('imgs/train_real_world/*/*.png')
jpgs = glob.glob('imgs/train_real_world/*/*.jpg')

files = pngs + jpgs

fake_features = np.array([[59,59,59],[102,102,102],[178,178,178],[0,255,0],[255,0,0]])



for file in files:
    img = cv2.imread(file)
    img, _, _, _, _, _  = seg.segment3(img)

    img= (img @ fake_features).astype(np.uint8)

    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    new_file = file.replace('train_real_world','train_real_world_segmented').replace('.jpg','.png')
    
    dir = os.path.dirname(new_file)

    os.makedirs(dir,exist_ok=True)

    cv2.imwrite(new_file,img)