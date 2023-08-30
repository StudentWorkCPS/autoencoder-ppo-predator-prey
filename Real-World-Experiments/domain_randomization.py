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
def repeat_pattern(pattern, tiles,rotate=True):
    result = np.zeros((pattern.shape[0] * tiles[0], pattern.shape[1] * tiles[1], 3))
    for i in range(tiles[0]):
        for j in range(tiles[1]):
            if rotate:  
                for k in range(np.random.randint(0, 4)):
                    pattern = np.rot90(pattern)

            result[i * pattern.shape[0]:(i + 1) * pattern.shape[0], j * pattern.shape[1]:(j + 1) * pattern.shape[1], :] = pattern
    
    return cv2.resize(result, (250, 250))

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
    for i in range(x):
        for j in range(y):
            noise[i,j] = np.random.uniform(0, 1,(c,))


    noise = cv2.resize(noise, (int(y * 10),int(x * 10)))
    noise = cv2.blur(noise, (13, 13))
    noise = cv2.dilate(noise, np.ones((5,5)))
    return noise

def voronoi(point_c):
    points = np.random.randint(0, 250, size=(point_c, 2))
    img = np.zeros((250, 250, 1))
    for i in range(250):
        for j in range(250):
            distances = []
            for point in points:
                distances.append(np.linalg.norm(np.array([i,j]) - point))
            img[i,j] = np.min(distances)
    return img

def add_noise(pattern, noise):
    return np.where(noise > 0.5, pattern, np.random.uniform(0, 1, size=pattern.shape))

def rand_i(min,max):
    return np.random.randint(min, max)

def color_swaping(pattern):
    pattern = pattern.copy()
    idx = list(range(3))

    np.random.shuffle(idx)

    for i in range(3):
        pattern[:,:,i] = pattern[:,:,idx[i]]
    
    return pattern

def crop(pattern):
    x = rand_i(0, 250 - 10)
    y = rand_i(0, 250 - 10)
    pattern = pattern[x:x+250, y:y+250]

    return cv2.resize(pattern, (250, 250))


op1 = lambda x: add_noise(x, perlin_noise(25,25,3))
op2 = lambda x: repeat_pattern(x,(rand_i(1,5),rand_i(1,5)))
operations = []#[ op1, op2, crop]

def create_fn(pattern):
    return lambda x: combine_patterns(x, pattern / 255)

operations = [create_fn(p) for p in patterns] + [crop, op1, op2]

def create_pattern():
    _pattern = np.ones((250, 250, 3)) * np.random.uniform(0, 1, size=(3,))
    #print(pattern.shape)
    #print(pattern.max())
    #pattern = combine_patterns([pattern, patterns[rand_i(0,len(patterns))]/ 255])
    for i in range(5):
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
    img = contrast(img, np.random.uniform(0.75, 1.25))
    img = brightness(img, np.random.uniform(-0.5, 0.5))
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

def segment(img):

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

    img = combine_dims(img, [2,5])

    return img


def apply_to_features(img, features):
    img = cv2.resize(img, (64,64), interpolation=cv2.INTER_NEAREST)
    result = img.copy() / 255
    for i,feature in enumerate(features):
        test_pattern = create_pattern()
        test_pattern = cv2.resize(test_pattern, (64,64))
    
        result = np.where(img == feature, test_pattern,result)

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