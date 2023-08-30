
import numpy as np 
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

import AutoEncoder.model_utils as mu
import cv2 

tf.config.set_visible_devices([], 'GPU')


class CAE(tf.keras.Model):
    def __init__(self, latent_dim, input_shape=(64,64,1),output_filters=1,segmentation=False,architecture='default'):
        super(CAE, self).__init__()
        self.latent_dim = latent_dim
        if architecture == 'default':
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=input_shape),
                    #tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(
                        filters=(32), kernel_size=3, strides=(2, 2), activation='relu'),
                    #tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(
                        filters=(64), kernel_size=3, strides=(2, 2), activation='relu'),
                    #tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Flatten(),
                    # No activation
                    tf.keras.layers.Dense(latent_dim),
                ],name='encoder')
            self.decoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                    tf.keras.layers.Dense(units=16*16*32, activation=tf.nn.relu),
                    tf.keras.layers.Reshape(target_shape=(16, 16, 32)),
                    #tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2DTranspose(
                        filters=64, kernel_size=3, strides=2, padding='same',
                        activation='relu'),
                    #tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2DTranspose(
                        filters=32, kernel_size=3, strides=2, padding='same',
                    ),
                    #tf.keras.layers.BatchNormalization(),
                    # No activation
                    tf.keras.layers.Conv2D(  
                        filters=output_filters, activation='sigmoid' if not segmentation else 'softmax', kernel_size=(3,3), strides=1, padding='same'),
                    #tf.keras.layers.BatchNormalization()
                ],name='decoder'
                )
            
        elif architecture == 'small':
            
            self.encoder = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=3, activation='relu'),

                tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim),
                ],name='encoder')
            
            self.decoder = tf.keras.Sequential([
                 tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                 tf.keras.layers.Dense(units=8 * 8 * latent_dim, activation='relu'),
                 tf.keras.layers.Reshape(target_shape=(8, 8, latent_dim)),
                 tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=4, padding='same',
                    activation='relu'),
                 tf.keras.layers.Conv2DTranspose(
                    filters=16, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                 tf.keras.layers.Conv2D(
                    filters=output_filters, kernel_size=3, activation='sigmoid' if not segmentation else 'softmax', padding='same'
                 )
                ],name='decoder')

            
        

        
    def encode(self, x,training=False):
        return self.encoder(x,training=training)
        
    def decode(self, z,training=False,apply_sigmoid=False):
        logits = self.decoder(z,training=training)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
        
    @tf.function
    def call(self, x):
        return self.decode(self.encode(x))
            
    #@tf.function
    #def compute_loss(self, x,y):
    #   return tf.keras.losses.MeanSquaredError()(x,self(x))
    
    @tf.function
    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False, layer_range=None):
        self.encoder.summary(line_length=line_length, positions=positions, print_fn=print_fn, expand_nested=expand_nested, show_trainable=show_trainable, layer_range=layer_range)
        self.decoder.summary(line_length=line_length, positions=positions, print_fn=print_fn, expand_nested=expand_nested, show_trainable=show_trainable, layer_range=layer_range)
    
    
    def load(self):
        self.load_weights('AutoEncoder/checkpoints/ae')
    
        
    def load(self):
        self.load_weights('AutoEncoder/checkpoints/ae')
    
    #@tf.function
    #def train_step(self, x,y):
    #    with tf.GradientTape() as tape:
    #        loss = self.compute_loss(x)
    #    grads = tape.gradient(loss, self.trainable_weights)
    #    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

    #        return loss



