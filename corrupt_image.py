import cv2
import numpy as np
import os
from random import randint



class Corrupter:
    def __init__(self,bright_value=(-75,75),
clip_limit=(3,6),title_grid_size=(8,12),blur_size=(2,4),
dir_x_images="images/X",dir_y_images="images/Y",dir_new_y_images="images/new_Y",
image_size=(512,512), tone_value=(-2,2), saturation_value=(-3,3)):
        self.bright_value=bright_value
        self.clip_limit=clip_limit
        self.title_grid_size=title_grid_size
        self.blur_size=blur_size
        self.valid_images = [".jpg",".gif",".png",".tga",".jpeg"]
        self.dir_x_images =dir_x_images
        self.dir_new_y_images =dir_new_y_images
        self.dir_y_images =dir_y_images
        self.image_size=image_size
        self.tone_value=tone_value
        self.saturation_value=saturation_value

    def inc_bright(self,img, value=30):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)

            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value

            final_hsv = cv2.merge((h, s, v))
            part_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

            return img

    def dec_bright(self,img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 0 + value
        v[v < lim] = 0
        v[v >= lim] -= value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
    
    def contrast(self,img,clip_limit=3.,title_grid_size=(8,8)):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=title_grid_size)

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
        l, a, b = cv2.split(lab)  # split on 3 different channels

        l2 = clahe.apply(l)  # apply CLAHE to the L-channel

        lab = cv2.merge((l2,a,b))  # merge channels
        img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
        return img2

    def noisy_ps(self,image,value=20):
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        if not (num_salt > 1):
            return image
        try:
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            out[coords] = 1

      # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            out[coords] = 0
            return out
        except Exception:
            return image

    def blur(self,img,blur_size=(5,5)):
        return cv2.blur(img,blur_size)

    def tone(self,img, value=10):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        img[:,:,0] =  img[:,:,0]  + value# Changes the V value

        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def Saturation(self,img, value=10):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        img[:,:,1] =  img[:,:,1]  + value# Changes the V value

        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img


    def part_axis_y(self):
        while True:
            self.y0=self.y_current
            if self.y0 == self.image_size[0]:
                break
            self.y1=self.y0+randint(self.min_size,self.max_size)
            if self.y1>self.image_size[0]:
                self.y1=self.image_size[0]
            bright_value=randint(self.bright_value[0],self.bright_value[1])
            blur_size=randint(self.blur_size[0],self.blur_size[1])
            tone_value=randint(self.tone_value[0],self.tone_value[1])
            clip_limit=randint(self.clip_limit[0],self.clip_limit[1])
            title_grid_size=randint(self.title_grid_size[0],self.title_grid_size[1])

            if bright_value > 0:
                part_image=self.inc_bright(self.cor_img[self.x0:self.x1,self.y0:self.y1],value=bright_value)
            else:
                part_image=self.dec_bright(self.cor_img[self.x0:self.x1,self.y0:self.y1],value=(bright_value*(-1)))
            
            part_image=self.contrast(part_image,clip_limit=clip_limit,
                title_grid_size=(title_grid_size,title_grid_size))   
            part_image=self.blur(part_image,blur_size=(blur_size,blur_size))
            
            part_image=self.tone(part_image,tone_value)
            
            # part_image=self.change_Saturation(part_image,saturation_value)
            
            part_image=self.noisy_ps(part_image)

            self.cor_img[self.x0:self.x1, self.y0:self.y1] = part_image
            self.y_current=self.y1
        self.x_current=self.x1
        self.y_current=0

    def part_axis_x(self):
        while True:
            self.x0=self.x_current
            if self.x0 == self.image_size[1]:
                break
            self.x1=self.x0+randint(self.min_size,self.max_size)            
            if self.x1>self.image_size[1]:
                self.x1=self.image_size[1]            
            self.part_axis_y()



    def corrupt(self,cor_img,parts=1):
        self.max_size=1.4*self.image_size[0]//parts
        self.min_size=0.5*self.image_size[0]//parts
        self.cor_img=cor_img
        self.x_current=0
        self.y_current=0        
        self.part_axis_x()            
        return self.cor_img

    def add_in_meta_file(self, filename):
        self.filewrite.write(filename)


    def generate_corrupt_images(self,count,parts=1):
        self.filewrite = open('images/kak_ygodno.txt', 'w')
        if not os.path.exists(self.dir_x_images):
             os.makedirs(self.dir_x_images)
        if not os.path.exists(self.dir_new_y_images):
             os.makedirs(self.dir_new_y_images)
        for i, file_y in enumerate(os.listdir(self.dir_y_images)):
            ext = os.path.splitext(file_y)[1]
            if ext.lower() not in self.valid_images:
                continue
            path_y_image = os.path.join(self.dir_y_images,file_y)
            imageName=str(i)+ext
            path_new_y_image=os.path.join(self.dir_new_y_images,imageName)
            img_y=cv2.imread(path_y_image)
            img_y=cv2.resize(img_y,self.image_size)
            cv2.imwrite(path_new_y_image, img_y)
            self.add_in_meta_file(imageName)
            for j in range(count):
                image=self.corrupt(np.copy(img_y),parts=parts) 
                file_x = imageName.replace(ext, "_"+str(j) + ext)
                path_x_image=os.path.join(self.dir_x_images,file_x)
                cv2.imwrite(path_x_image, image)
                self.add_in_meta_file(" " + file_x)
            self.filewrite.write("\n\r")
        self.filewrite.close()

if __name__ == "__main__":
	myclass=Corrupter()
	myclass.generate_corrupt_images(40,parts=4)
