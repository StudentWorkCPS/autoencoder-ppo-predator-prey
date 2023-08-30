import domain_randomization as dr

import glob
import tensorflow as tf
import numpy as np
import cv2

import segment as seg

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


prey = np.eye(5)[3]
predator = np.eye(5)[4]
# enable tf numpy ops
def preporcess_image(file,mode,real_img=False):
    file = bytes.decode(file)
    mode = bytes.decode(mode)
    load_img = dr.load_image(file)

    img = load_img.copy()
    label = load_img.copy()

    if mode == 'segment':

        # learn a more robust fov-model
        if np.random.random() > 0.5:
            load_img = cv2.flip(load_img,1)
            pass
        
        img = cv2.resize(load_img,(64,64),interpolation=cv2.INTER_NEAREST)

        if np.random.random() < 0.1:
            w,h = np.random.randint(62,66),np.random.randint(62,66)
            
            img = cv2.resize(img,(w,h),interpolation=cv2.INTER_NEAREST)

            img = cv2.resize(img,(64,64),interpolation=cv2.INTER_NEAREST)

        if np.random.random() < 0.1:
            img = cv2.blur(img,(5,5))

        #img = dr.blob_fill(img,dr.predator)[0]
        #img = dr.blob_fill(img,dr.prey)[0]
        img = dr.segment(img)
        img = dr.blob_fill(img,predator)[0]
        img = dr.blob_fill(img,prey)[0]
       
 
        label = img


    elif mode == 'resize':
        img = cv2.resize(img,(64,64),interpolation=cv2.INTER_NEAREST)
        
        label = dr.segment(img)

    elif mode == 'domain_randomization':

        if np.random.random() < 0.1:
            w,h = np.random.randint(60,68),np.random.randint(64,68)
            
            img = cv2.resize(img,(w,h),interpolation=cv2.INTER_NEAREST)

            img = cv2.resize(img,(64,64),interpolation=cv2.INTER_NEAREST)
    
        if np.random.random() < 0.01:
            img = cv2.blur(img,(5,5))


        img = dr.domain_random(load_img) / 255

        load_img = cv2.resize(load_img,(64,64))
        label = dr.segment(load_img,combine=False)

        if np.random.random() > 0.5:
            img = np.fliplr(img)
            label = np.fliplr(label)

        



    elif mode == 'domain_randomization_segment':
        pass

    if real_img:
        return img, label, cv2.resize(load_img,(64,64),interpolation=cv2.INTER_NEAREST)

    return img, label



def create_dataset(files,batch_size=32,epchos=100,options={
    'mode':'segment',
    'real_img':False
}):
    print('Creating dataset with mode: ' + options['mode'])
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.shuffle(buffer_size=len(files)) 
    numpy_wrap = lambda x: x
    if options['real_img']:
        numpy_wrap = lambda x: tf.numpy_function(func=preporcess_image, inp=[x,options['mode'],options['real_img']], Tout=(tf.double,tf.double,tf.uint8))
    else:
        numpy_wrap = lambda x: tf.numpy_function(func=preporcess_image, inp=[x,options['mode']], Tout=(tf.double,tf.double))

    dataset = dataset.map(numpy_wrap, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(epchos)

    # Create an iterator for the dataset
    iterator = iter(dataset)

    return iterator


def create(path,batch_size=32,epchos=100,options={'mode':'segment'}):
    files = glob.glob(path)

    assert len(files) > 0, 'No files found in path: ' + path

    shuffle = np.random.permutation(len(files))
    files = np.array(files)[shuffle]

    test_fs = files[:int(len(files)*0.2)]
    train_fs = files[int(len(files)*0.2):]

    train = create_dataset(train_fs,batch_size,epchos,options)
    test = create_dataset(test_fs,batch_size,epchos,options)

    return train,test, len(train_fs),len(test_fs)

    
if __name__ == '__main__':

    other,it,img_len,label_len = create('./data/trutlebot4-0/*/*.png',6,epchos=1,options={'mode':'segment','real_img':True})

    imgs,labels,real = next(it)
    print(imgs.shape)
    print('Label',labels.shape)
    #print(batch.dtype)

    labels = labels @ np.array(dr.all_features[:-1])
    labels = labels.reshape(-1,64,64,3)
    labels = np.concatenate(labels,axis=1)


    imgs = imgs @ np.array(dr.all_features[:-1])
    imgs = imgs.reshape(-1,64,64,3)
    imgs = np.concatenate(imgs,axis=1)

    real = np.concatenate(real,axis=1)


    #load_img = np.concatenate(load_img,axis=1)

    print(labels.shape)
    
    print('Label',labels.shape)
    print('Images',imgs.shape)
    print('Real',real.shape)
    #labels = np.concatenate([labels,labels,labels],axis=2)
    #result = np.concatenate([imgs,labels,load_img],axis=0).astype(np.uint8)
    result = np.concatenate([imgs,labels,real],axis=0).astype(np.uint8)


    result = cv2.cvtColor(result , cv2.COLOR_RGB2BGR)
    result = cv2.resize(result,(result.shape[1]*3,result.shape[0]*3))
    cv2.imshow('img',result)

    cv2.waitKey(0)



