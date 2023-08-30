import matplotlib.pyplot as plt
import glob
import os 
import shutil
import io
import tensorflow as tf
import random
import cv2
import numpy as np

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

STANDARD_SAVE_PATH = 'model.ckpt'

def generate_and_save_images(model,path, test_sample):
  predictions = model.decoder(test_sample)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, :],cmap='gray')
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  #plt.savefig(path)

  return fig


def plot_to_image(figure):
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  plt.close(figure)
  buf.seek(0)
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  image = tf.expand_dims(image,0)

  return image

def clean_epochs():
  # delete dir if exists
  if os.path.exists('epochs'):
    shutil.rmtree('epochs')

  
  os.mkdir('epochs')

def summarize_model(model,name='model'):
    text = '{}\n'.format(name)
    def add_line(x):
        nonlocal text
        text += x + '\n'

    model.summary(print_fn=add_line)

    return text


def save_model(model,options,save_path=STANDARD_SAVE_PATH):
    os.mkdir(save_path)

    info_path = save_path+os.sep+'info.txt'
    print(info_path)
    with open(info_path, 'w') as f:
      f.write("[options]\n")
      for key in options.keys():
          f.write("{}={}".format(key,options[key]))
          f.write('\n')
      f.write('\n[save]\n')
      f.write("save_weights=%s\n"%save_path)
      f.write("\n[model]\n")    
      model.summary(print_fn=lambda x: f.write(x + '\n'))

    shutil.move('prediction.png',save_path+os.sep+'prediction.png')
    #shutil.move('losses.png',save_path+os.sep+'losses.png')
    #shutil.move('epochs',save_path+os.sep+'epochs')
    #shutil.move('logs',save_path+os.sep+'logs')

    shutil.move('test_samples.png',save_path+os.sep+'test_sampe.png')
    model.save_weights(save_path+os.sep+'model.ckpt')


def load_model(ModelClass,latent_dim,save_path=STANDARD_SAVE_PATH):
    model = ModelClass(latent_dim)
    model.load_weights(save_path)
    return model


def load_data(current_dataset):
    files = glob.glob(f'data/{current_dataset}/*.png')
    data = []
    for file in files:
      img = cv2.imread(file)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = cv2.resize(img, (64, 64))
      data.append(img)

    random.shuffle(data)

    return np.array(data) 

def normalize(images):
  images = images / 255
  return np.reshape(images, [-1,64, 64, 3]).astype('float32')

def grey_scale(images,colors):
    images = np.dot(images[...,:3], colors)
    images = np.reshape(images, [-1,64, 64, 1]).astype('float32')
    return images

def preprocess_img(img):
    img = normalize(img)
    img = grey_scale(img,[1,-0.5,0])

    return img