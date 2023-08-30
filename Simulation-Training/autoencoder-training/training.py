import numpy as np 
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

import real_world as rw

import glob
import cv2 
import random

import datetime
import model_utils

import dataset

import argparse
import json

from ae import CAE





args = argparse.ArgumentParser()
args.add_argument('--dataset',type=str,default='data/trutlebot4-0/*/*.png')
args.add_argument('--epochs',type=int,default=100)
args.add_argument('--batch_size',type=int,default=32)
args.add_argument('--latent_dim',type=int,default=32)
args.add_argument('--lr',type=float,default=0.0001)
args.add_argument('--load-checkpoint',type=str,default=None)
args.add_argument('--mode',type=str,default='segment')
args.add_argument('--architecture',type=str,default='default')
args.add_argument('--use-lr-decay',action='store_true')
args.add_argument('--freeze-layers',type=str,default=None,help='comma separated list of layers ["encoder","decoder"] to freeze')

args = args.parse_args()

'''
files = glob.glob(f'data/{args.dataset}/*.png')
data = []
for file in files:
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))
    data.append(img)

random.shuffle(data)
'''
'''
def preprocess_images(images):
  images = np.array(images) / 255
  return np.reshape(images, [-1,64, 64, 3]).astype('float32')

def grey_scale(images,colors):
    images = np.dot(images[...,:3], colors)
    images = np.reshape(images, [-1,64, 64, 1]).astype('float32')
    return images
'''
  
'''
data = preprocess_images(data)
grey_data = grey_scale(data,[1, -0.5, 0])
 
train,test = grey_data[:int(len(grey_data)*0.8)],grey_data[int(len(grey_data)*0.8):]
print("got data (train:",train.shape,' and test: ',test.shape,')')
'''

# Training Data
train_iter,test_iter,train_size,test_size = dataset.create(f'{args.dataset}',args.batch_size,args.epochs, {'mode':args.mode,'real_img':False})


test_sample = next(test_iter)[0]

lr = args.lr

if args.use_lr_decay:
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.lr,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True
    )

optimizer = tf.keras.optimizers.legacy.Adam(lr)

print('Batch Shape: ',test_sample.shape)

features = None
if args.mode == 'domain_randomization':
    features = dataset.dr.all_features
else :
    features = dataset.dr.all_features[:-1]

print("Latent Dim: ",args.latent_dim)
print("Input Shape",test_sample.shape[1:])
print("Output Filters",len(features))

model = CAE(
            args.latent_dim,
            input_shape=test_sample.shape[1:],
            output_filters=len(features),
            segmentation=True,
            architecture='small',
            latent_activation=None,
            freeze=json.loads( args.freeze_layers) if args.freeze_layers is not None else None
        )

#
model.compile(optimizer=optimizer,loss='categorical_crossentropy')

if args.load_checkpoint:
    model.load_weights(args.load_checkpoint)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
file_writer_cm = tf.summary.create_file_writer(log_dir + "/cm")

#test_sample = test[:1]

def result2img(x):
    #print("draw Image")
    #print(x.shape)
    #print(features)
    return (x @ features).astype('uint8')


def log_confusion_matrix(epoch, logs):
    # Use the model to predict the values from the validation dataset.
    fig = model_utils.generate_and_save_images(model, epoch, model.encode(test_sample), result2img=result2img)
    img = model_utils.plot_to_image(fig)

    

    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", img, step=epoch)

        if args.mode == 'domain_randomization':
            latent_loss, reconstruction_loss = rw.compute_real_world_sim(model)
            tf.summary.scalar('rw latent loss', latent_loss, step=epoch)
            tf.summary.scalar('rw reconstruction loss', reconstruction_loss, step=epoch)
        if args.use_lr_decay:
            tf.summary.scalar('lr', optimizer.lr(epoch), step=epoch)
        else:
            tf.summary.scalar('lr', optimizer.lr, step=epoch)


def show_predictions(epoch, logs):
    # Use the model to predict the values from the validation dataset.
    fig = plt.figure(figsize=(4, 4))
    

    real_img = test_sample[0]
    

    prediction_tensor = model(real_img.reshape(1,64,64,-1))
    #real_img = result2img(real_img)
    if real_img.shape[-1] > 3:
        real_img = result2img(real_img)

    real_img = real_img.numpy()
    prediction = prediction_tensor.numpy()
    prediction = result2img(prediction)


    plt.subplot(1, 2, 1)
    plt.imshow(real_img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(prediction.reshape(64,64,-1), cmap='gray')

    img = model_utils.plot_to_image(fig)

    with file_writer_cm.as_default():
        tf.summary.image("Predictions", img, step=epoch)



#cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
pred_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=show_predictions)

checkpoint_dir = f'./checkpoints/ae-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


model.summary()

print('Training...')
print('Train Size: ',train_size,' Test Size: ',test_size)
model.fit(train_iter,
          epochs=args.epochs,
          batch_size=args.batch_size,
          steps_per_epoch=train_size//args.batch_size,
          
          #validation_data=test_iter,
          validation_data=test_iter,
          validation_steps=test_size//args.batch_size,
          callbacks=[tensorboard_callback,cm_callback,pred_callback,model_checkpoint_callback]
          )


#model.save_weights(f'checkpoints/ae-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')