import cv2
import numpy as np
from PIL import ImageFont

def norm(x):
    max = np.max(x)
    min = np.min(x)
    return (x - min) / (max - min)

def lerp(a,b,t):

    return a + t * (b - a)

def count_color(img,color):
    img = img.reshape(-1,3)
    return np.sum(np.all(img == color,axis=1))


text_color = (255,255,255)
font_size = 1
font_path = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf"

# Load the font using ImageFont.truetype()
font = cv2.FONT_HERSHEY_PLAIN


def debug(image,infos, start_point=(5,15)):
    for i,info in enumerate(infos):
        x = start_point[0]
        y = start_point[1] + 15 * (i)

        image = cv2.putText(image,info,(x,y),font,font_size,text_color,thickness=1)
    return image

def plot(x,y,size,normalize=True):
    w,h = size

    if normalize:
        x = norm(x)
        y = norm(y)
    
    #print(y.shape)
    img = np.ones((w,h,3), np.uint8)*255
    l = len(x)
    for i in range(l):

        start_point = (int(x[i] * w),int(h / 2))
        end_point = (int((x[i+1] if i+1 < l else 1) * w) ,int(h/2 - y[i] * h/2))
        color = lerp(np.array((0,0,1)),np.array((1,0,0)),x[i])
  
        color = (int(color[0] * 255),int(color[1] * 255),int(color[2] * 255))

        
        img = cv2.rectangle(img,start_point,end_point, color, -1) 

    return img / 255

def draw_hor(image,imgs):
    for img in imgs:
        image = np.concatenate((image,img),axis=1)
    
    return image

def draw_ver(image,imgs):
    for img in imgs:
        image = np.concatenate((image,img),axis=0)
    
    return image

def grey_to_rgb(img):
    return np.repeat(img, 3, axis=2)

length = np.vectorize(len)

def text_to_length(text,length):
    if len(text) > length:
        return text[:length]
    return text + ' ' * (length - len(text))
    

def debug_table(img,infos,start_point=(5,15)):    
    lengths = length(infos)

    ls = np.max(lengths,axis=0)


    for i,info in enumerate(infos):
        text = [text_to_length(text,l) for text,l in zip(info,ls)]
        text = '|'.join(text)
        x = start_point[0]
        y = start_point[1] + 15 * i 

        img = cv2.putText(img,text,(x,y),font,font_size,text_color,1)

        
    return img

latent_max = 0
latent_min = 0

def draw_robot_view(normal_img,preprocced_img,predicted_img,latent):
    global latent_max,latent_min


    # To Be able to display
    #preprocced_img = grey_to_rgb(preprocced_img[0])
    #predicted_img = grey_to_rgb(predicted_img[0])

    # Simulate Matplotlib grey plot
    #preprocced_img = norm(preprocced_img) 
    preprocced_img = preprocced_img / 255
    predicted_img = predicted_img / 255
    
    #predicted_img = norm(predicted_img)

    # Plot latent 'Activations'
    if not (type(latent) == np.ndarray):
        latent = latent.numpy()
    
    latent = latent[0]
        
    latent_max = max(np.max(latent),latent_max)
    latent_min = - latent_max
    
    zero =  -latent_min / (latent_max - latent_min)

    latent = ((latent - latent_min) / (latent_max - latent_min) - zero )
    latent_plot = plot(np.arange(32)/32,latent,(64,64),normalize=False)
    

    # Add Zero Line for Reference
    latent_plot = cv2.line(latent_plot,(0,int(zero * 64)),(64,int(zero * 64)),(0,0,0),1)
    
    # Draw Images
    img = draw_hor(normal_img/255,[preprocced_img,predicted_img,latent_plot])
    
    return img



class CV2View():
    def __init__(self,window_name='Debug'):
        self.window_name = window_name
        self.image = np.array([])
        self.current_line_c = 0

    def rectangle(self,start_point,end_point,color,thickness):
        self.image = cv2.rectangle(self.image,start_point,end_point,color,thickness)

        return self
    
    def line(self,start_point,end_point,color,thickness):
        self.image = cv2.line(self.image,start_point,end_point,color,thickness)

        return self

    def circle(self,center,radius,color,thickness):
        self.image = cv2.circle(self.image,center,radius,color,thickness)

        return self
    
    def resize(self,h,w):
        self.image = cv2.resize(self.image,(h,w))

        return self

        
    def debug(self,infos):
        self.image = debug(self.image,infos,start_point=(5,15 + 15 * self.current_line_c))
        self.current_line_c += len(infos)

        return self

    def debug_table(self,infos):
        self.image = debug_table(self.image,infos,start_point=(5,15 + 15 * self.current_line_c))
        self.current_line_c += len(infos)
        return self

    def draw_robot_view(self,normal_img,preprocced_img,predicted_img,latent):
        print(normal_img.shape,preprocced_img.shape,predicted_img.shape,latent.shape)
        img = draw_robot_view(normal_img,preprocced_img,predicted_img,latent)
        self.image = draw_ver(self.image,[img]) if self.image.size else img

        return self


    def can_show(self):
        return self.image.size

    def show(self):
        #print(self.image.shape)
        if not self.image.size:
            return

        self.image = self.image.astype(np.float32)

        self.image = cv2.cvtColor(self.image,cv2.COLOR_RGB2BGR)
        
        cv2.imshow(self.window_name,self.image)

        self.current_line_c = 0
        self.image = np.array([])


    def close(self):
        cv2.destroyAllWindows()