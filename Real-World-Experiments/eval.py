import segment2 as seg
import glob as glob
import AutoEncoder.ae as ae
import numpy as np
import tensorflow as tf

import cv2

fake_features = np.array([[59,59,59],[102,102,102],[178,178,178],[0,255,0],[255,0,0]]) 

def load_models():
    paths = glob.glob("EvalAEs/*/*.index")
    models = []
    print(paths)
    for path in paths:
        model = ae.CAE(32,(64,64,5),5,True,'small')
        model.load_weights(path[:-6])
        models.append(model)
    return models

def load_imgs():

    # Load images
    imgs = []
    segmented_imgs = []
    for img_path in glob.glob("imgs/train_real_world/*/*.jpg"):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        segmented_imgs.append(np.array(seg.segment3(img)[0]))



    return imgs,segmented_imgs

#cross_entropy = np.vectorize(cross_entropy_fn)


models = load_models()
imgs,segmented_imgs = load_imgs()
#idxs = np.random.randint(0,len(imgs),5)
cce = tf.keras.losses.CategoricalCrossentropy()
print(len(imgs))

total_loss = np.array([0,0,0]).astype(np.float32)


for i in range(len(imgs)):
    img = cv2.resize(imgs[i],(64,64))
    recons = [ model(segmented_imgs[i].reshape(1,64,64,-1)) for model in models]
    loss = [cce(segmented_imgs[i].reshape(1,64,64,-1),recon).numpy() for recon in recons]
    total_loss += np.array(loss)

print(total_loss/len(imgs))


