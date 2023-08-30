import domain_randomization as dr
import numpy as np
import cv2
import glob
import os

real = glob.glob('data/real_world_test/real_world/*.png')
segmented = glob.glob('data/real_world_test/segmented/*.png')

real_to_segmented = {}

for file in real:
    name = file.split('/')[-1]
    name = name.split('.')[0]

    real_to_segmented[name] = (file,None)

for file in segmented:
    name = file.split('/')[-1]
    name = name.split('.')[0]
    real_to_segmented[name] = (real_to_segmented[name][0],file)


def compute_real_world_sim(model):

    latent_loss = np.zeros((len(real_to_segmented)))
    reconstruction_loss = np.zeros((len(real_to_segmented)))
    for i,(name,(real,segmented)) in enumerate(real_to_segmented.items()):
        if segmented is None:
            continue

        img = cv2.imread(real)
        img = cv2.resize(img,(64,64))
        img = img / 255

        fake_img = cv2.imread(segmented)
        fake_recon = dr.segment(fake_img,False)
        #fake_img = cv2.resize(fake_img,(64,64))
        fake_img = dr.domain_random(fake_img) / 255   


        latent0 = model.encode(img.reshape((1,64,64,3)))
        latent1 = model.encode(fake_img.reshape((1,64,64,3)))

        latent_loss[i] = np.mean((latent0-latent1) ** 2,axis=1)

        reconstruction0 = model.decode(latent0)

        # Categorical cross entropy
        reconstruction_loss[i] = categorical_cross_entropy(fake_recon,reconstruction0)

    return latent_loss.mean(),reconstruction_loss.mean()

def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-10  # small value to avoid division by zero

    # Clip the predicted values to prevent NaNs in the logarithm
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

    # Compute the cross-entropy
    cross_entropy = -np.sum(y_true * np.log(y_pred))

    return cross_entropy


if __name__ == '__main__':
    import ae as ae

    model = ae.CAE(32)

    model.load_weights('checkpoints/ae-20230628-091928')

    print(compute_real_world_sim(model))

    
