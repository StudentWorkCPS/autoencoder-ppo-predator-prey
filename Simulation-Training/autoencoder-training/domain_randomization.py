import numpy as np

import cv2
from glob import glob

# Debug 
import matplotlib.pyplot as plt



def insta(img):

    width = img.shape[1]
    height = img.shape[0]

    min = np.min([width,height])
    x = width // 2 - min // 2
    y = height // 2 - min // 2
    
    #print(x,y,min)
    img = img[y:y+min, x:x+min, :]
    return cv2.resize(img, (250, 250),interpolation=cv2.INTER_NEAREST)

def load_images(path):
    files = glob(path)
    images = []
    for file in files:
        img = load_image(file)
        images.append(img)
    
    return np.array(images)

def load_image(file):
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = insta(image)
    return image

patterns = load_images('patterns/*.png')
patterns = np.array([cv2.resize(pattern, (64, 64)) for pattern in patterns])

def repeat_pattern(pattern, tiles,rotate=True):
    result = np.zeros((pattern.shape[0] * tiles[0], pattern.shape[1] * tiles[1], 3))
    for i in range(tiles[0]):
        for j in range(tiles[1]):
            if rotate:  
                for k in range(np.random.randint(0, 4)):
                    pattern = np.rot90(pattern)

            result[i * pattern.shape[0]:(i + 1) * pattern.shape[0], j * pattern.shape[1]:(j + 1) * pattern.shape[1], :] = pattern
    
    return cv2.resize(result, (64, 64))

def random_colors(pattern):
    return pattern * np.random.uniform(3) * 0.5 + np.random.rand(3) * 0.5

def blur_color(pattern):
    weights = np.random.uniform(0, 10, size=(3,3))
    weights = weights / np.sum(weights)
    return cv2.filter2D(pattern, -1, weights)



def combine_patterns(s,p):
    
    filter = p != 0 
    result = np.where(filter, p, s)

    return result

def perlin_noise(x,y,c):

    noise = np.zeros((x,y,c))
    
    noise = np.random.uniform(0, 1,(x,y,c))

    noise = cv2.resize(noise, (int(y * 5),int(x * 5)))
    noise = cv2.blur(noise, (13, 13))
    #noise = cv2.dilate(noise, np.ones((5,5)))

    noise = cv2.resize(noise, (64,64))
    return noise

def voronoi(point_c):
    points = np.random.randint(0, 250, size=(point_c, 2))
    img = np.zeros((64, 64, 1))
    for i in range(64):
        for j in range(64):
            distances = []
            for point in points:
                distances.append(np.linalg.norm(np.array([i,j]) - point))
            img[i,j] = np.min(distances)
    return img

def add_noise(pattern, noise):
    return np.where(noise > 0.5, pattern, np.zeros((64,64,1)) * np.random.uniform(0, 1, size=(3,)))

def rand_i(min,max):
    return np.random.randint(min, max)

def color_swaping(pattern):
    pattern = pattern.copy()
    idx = [0,1,2]

    np.random.shuffle(idx)

    for i in [0,1,2]:
        pattern[:,:,i] = pattern[:,:,idx[i]]
    
    return pattern

def crop(pattern):
    x = rand_i(0, 64 - 10)
    y = rand_i(0, 64 - 10)
    pattern = pattern[x:x+64, y:y+64]

    return cv2.resize(pattern, (64, 64))


op1 = lambda x: add_noise(x, perlin_noise(10,10,3))
op2 = lambda x: repeat_pattern(x,(rand_i(2,3),rand_i(2,)))
operations = []#[ op1, op2, crop]

def create_fn(pattern):
    return lambda x: combine_patterns(x, pattern / 255)

operations = [create_fn(p) for p in patterns] + [op1,crop] #+ [crop, op1, op2]

def create_pattern():
    _pattern = np.ones((64, 64, 3)) * np.random.uniform(0, 1, size=(3,))
    #print(pattern.shape)
    #print(pattern.max())
    #pattern = combine_patterns([pattern, patterns[rand_i(0,len(patterns))]/ 255])
    for i in range(2):
        _op = operations[rand_i(0,len(operations))]
        _pattern = _op(_pattern)
        #print(_pattern.max())

    _pattern = color_swaping(_pattern)
    _pattern = random_colors(_pattern)
    #pattern = blur_color(pattern)

    return _pattern


def contrast(img, contrast):
    return (img - 0.5) * contrast + 0.5

def brightness(img, brightness):
    return img + brightness

def saturation(img, saturation):
    return img * saturation

def hue(img, hue):
    return img + hue

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def randomize(img):
    img = contrast(img, np.random.uniform(0.85, 1.15))
    img = brightness(img, np.random.uniform(-0.1, 0.1))
    img = saturation(img, np.random.uniform(0.9, 1.1))
    #img = hue(img, np.random.uniform(-0.1, 0.1))
    max = np.max([np.max(img),1])
    return np.clip(img,0,max) / max

ground = [59, 59, 59]
wall = [102, 102, 102]
sky = [178,178,178]

prey = [1,104,1]
predator = [104,0,0] 
robot = [50,50,50]


features = [ground, wall,sky]

all_features = [ground, wall,sky, prey, predator,robot]

def one_hot(img, features):
    img = img.reshape(-1,3)
    img = np.array(np.argmin([np.sum((img - feature)**2, axis=1) for feature in features], axis=0))
    #print(img.shape, img.max(), img.min())
    img = np.eye(len(features))[img]

    img = img.reshape(64,64,-1)
    
    return img

def randomize_one_hot(img,combine=True):
    parameter = np.random.randint(0, len(all_features))

    img = one_hot(img, all_features)

    w = np.random.randint(1, 64)
    h = np.random.randint(1, 64)
    x = np.random.randint(0, 64 - w)
    y = np.random.randint(0, 64 - h)

    mask = (img[x:x+w, y:y+h, parameter] == 1)
    #print(mask.shape)
    mask = np.repeat(mask.reshape(w,h,1), len(all_features), axis=2)
    #print(mask)
    other = np.random.randint(0, len(all_features))
    other_one_hot = np.eye(len(all_features))[other]

    img[x:x+w, y:y+h] = np.where(mask, other_one_hot, img[x:x+w, y:y+h])

    if combine:
        img = combine_dims(img, [2,5])

    return img

def in_bound(x, min, max):
    return min <= x <= max

def blob_detection(img,feature):
    
    in_range = img == feature
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
    blobs = list(filter(lambda x: len(x['points']) > 3),blobs)
    #print('BLOBS:',blobs)

    return blobs,in_range

def blob_detection2(img,feature):
    #print(feature)
    img = (img == feature).all(axis=2)
    arr = np.argwhere(img)
    #print("ARRAX IDXS",arr)
    blobs = []
    for j in range(0,len(arr)):
        for i in range(0,len(blobs)):
            
            x,y = arr[j]
            blob = blobs[i]
            if in_bound(x,blob['min'][0] - 4,blob['max'][0] + 4) and in_bound(y,blob['min'][1] - 4,blob['max'][1] + 4):
                blob['min'] = np.min([[x,y],blob['min']],axis=0)
                blob['max'] = np.max([[x,y],blob['max']],axis=0)
                break
        else:
            blobs.append({
                'min': arr[j],
                'max': arr[j],
            })

    #print("BEFOR FILTERING:",len(blobs))
    return blobs,img

    


def blob_fill(segm,feature):
    blobs,in_range = blob_detection2(segm,feature)
    mask = np.zeros((64,64,1),np.uint8)
    #print('BLOBS:',blobs)
    for i in range(0,len(blobs)):
        blob = blobs[i]
        min = blob['min']
        max = blob['max']
        
        segm[min[0]:max[0],min[1]:max[1]] = feature
        #= cv2.rectangle(img,(min[1],min[0]),(max[1],max[0]),feature,-1)
        mask[min[0]:max[0],min[1]:max[1],0] = 1
        #print('BLOB:',min,max,feature.color)

    return segm,in_range,mask



def segment(img, combine=True):

    img = cv2.resize(img, (64,64), interpolation=cv2.INTER_NEAREST)
    #print(np.unique(img.reshape(-1,3), axis=0))
    img = img.astype(np.uint8)
    #img = img * (edges == 0).reshape(64,64,1)
    #img = img.astype(np.uint8)
    #mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2, 1)).astype(np.uint8)

    #for i,feature in enumerate(all_features):

    #    args = np.argwhere((img == feature).all(axis=2))
        
    #    print(args.shape)
    #    for arg in args:
    #        rgb = (int(feature[0]), int(feature[1]), int(feature[2]))
    #        _,img,mask,_ = cv2.floodFill(img, mask, (arg[1], arg[0]), rgb,(5,5,5),(5,5,5))

    img = one_hot(img, all_features)
    
    if combine:
        img = combine_dims(img, [2,5])

    return img


def apply_to_features(img, features):
    img = cv2.resize(img, (64,64), interpolation=cv2.INTER_NEAREST)
    result = img.copy() / 255
    for i,feature in enumerate(features):
        test_pattern = create_pattern()
        #test_pattern = cv2.resize(test_pattern, (64,64))
        mask = np.all(img == feature, axis=2)
        result[mask] = test_pattern[mask]

        #result = img == feature * test_pattern) result

    return result

def combine_dims(img,dims):

    # [[[1,0,0],[0,1,0]],[[0,0,1],[0,1,0]]] -0,2-> [[[1,0],[0,1]],[[1,0],[0,1]]]
    # [[[1,0,0],[0,1,0]],[[0,0,1],[0,1,0]]] -1,2-> [[[1,0],[0,1]],[[0,1],[0,1]]]
    # [[[1,0,0],[0,1,0]],[[0,0,1],[0,1,0]]] -0,1-> [[[1,0],[0,1]],[[1,0],[0,1]]]
    img = img.copy()
    min_dim = min(dims)
    idx = 0
    for dim in dims:
        real_dim = dim - idx
        real_min_dim = min_dim - idx
        if dim != min_dim:
            img[:,:,real_min_dim] = np.clip(img[:,:,real_min_dim] + img[:,:,real_dim],0,1)
            img = np.delete(img, real_dim, axis=2)
            idx = idx + 1

    return img

        



def domain_random(img):
    img = apply_to_features(img, features)

    img = randomize(img)

    return (img * 255).astype(np.uint8)