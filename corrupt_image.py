# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from random import randint


class Corrupter:
    def __init__(self,brightness_value=(-75,75),
clipLimit=(3,6),titleGridSize=(8,12),
gauss_value=(5,150),blur_size=(2,4),
dir_x_images="images/X",dir_y_images="images/Y",dir_new_y_images="images/new_Y",
image_size=(512,512), tone_value=(-2,2), saturation_value=(-3,3)):
        self.brightness_value=brightness_value
        self.clipLimit=clipLimit
        self.titleGridSize=titleGridSize
        self.gauss_value=gauss_value
        self.blur_size=blur_size
        self.valid_images = [".jpg",".gif",".png",".tga",".jpeg"]
        self.dir_x_images =dir_x_images
        self.dir_new_y_images =dir_new_y_images
        self.dir_y_images =dir_y_images
        self.image_size=image_size
        self.tone_value=tone_value
        self.saturation_value=saturation_value

    def increase_brightness(self,img, value=30):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)

            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value

            final_hsv = cv2.merge((h, s, v))
            part_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

            return img

    def decrease_brightness(self,img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 0 + value
        v[v < lim] = 0
        v[v >= lim] -= value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
    
    def change_contrast(self,img,clipLimit=3.,titleGridSize=(8,8)):
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=titleGridSize)

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
        # print(image.shape)
        num_salt = np.ceil(amount * image.size * s_vs_p)
        if not (num_salt > 1):
            return image
        # print(num_salt)
        try:
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            out[coords] = 1

      # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            out[coords] = 0
            return out
        except Exception:
            print(image.shape)
            print("ERROR, WHYYYY??")
            return image
        # row,col,ch= image.shape
        # mean = 0
        # var = 5 #max=2500
        # sigma = var**0.5
        # gauss = np.random.normal(mean,sigma,(row,col,ch))
        # gauss = gauss.reshape(row,col,ch)
        # noisy = image + gauss
        # return noisy
    def blur_image(self,img,blur_size=(5,5)):
        return cv2.blur(img,blur_size)

    def change_tone(self,img, value=10):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        img[:,:,0] =  img[:,:,0]  + value# Changes the V value
        #print(img[:,:,0])
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def change_Saturation(self,img, value=10):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        img[:,:,1] =  img[:,:,1]  + value# Changes the V value
        #print(img[:,:,0])
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def corrupt(self,cor_img,parts=1):
        # cor_img=cv2.imread(img)
        # cor_img=cv2.resize(cor_img,self.image_size)
        random_max=1.4*self.image_size[0]//parts
        random_min=0.5*self.image_size[0]//parts
        x_current=0
        y_current=0
        while True:            
            #x0=(i*cor_img.shape[0])//parts
            #x1=((i+1)*cor_img.shape[0])//parts
            x0=x_current
            if x0 == self.image_size[1]:
            	break
            x1=x0+randint(random_min,random_max)            
            if x1>self.image_size[1]:
            	#print("x0-"+str(x0))
            	#print("x1-"+str(x1))
            	x1=self.image_size[1]
            while True: 
                #print(img.shape)

                #y0=(o*cor_img.shape[1])//parts
                #y1=((o+1)*cor_img.shape[1])//parts
                y0=y_current
                if y0 == self.image_size[0]:
                	break
                y1=y0+randint(random_min,random_max)
                if y1>self.image_size[0]:
                	#print("y0-"+str(y0))
                	#print("y1-"+str(y1))
                	y1=self.image_size[0]
                brightness_value=randint(self.brightness_value[0],self.brightness_value[1])
                #brightness_value=randint(0,1)
                saturation_value=randint(self.saturation_value[0],self.saturation_value[1])
                blur_size=randint(self.blur_size[0],self.blur_size[1])
                tone_value=randint(self.tone_value[0],self.tone_value[1])
                gauss_value=randint(self.gauss_value[0],self.gauss_value[1])
                clipLimit=randint(self.clipLimit[0],self.clipLimit[1])
                titleGridSize=randint(self.titleGridSize[0],self.titleGridSize[1])

                #print("bright="+str(brightness_value))
                if brightness_value > 0:
                    part_image=self.increase_brightness(cor_img[x0:x1,y0:y1],value=brightness_value)
                else:
                    part_image=self.decrease_brightness(cor_img[x0:x1,y0:y1],value=(brightness_value*(-1)))
                
  
                part_image=self.change_contrast(part_image,clipLimit=clipLimit,
titleGridSize=(titleGridSize,titleGridSize))   
                part_image=self.blur_image(part_image,blur_size=(blur_size,blur_size))
                
                part_image=self.change_tone(part_image,tone_value)
                
                # part_image=self.change_Saturation(part_image,saturation_value)

                part_image=self.noisy_ps(part_image)

                for j in range(part_image.shape[0]):
                        for z in range(part_image.shape[1]):
                            cor_img[x0+j,y0+z]=part_image[j,z]
                y_current=y1
            x_current=x1
            y_current=0    
        return cor_img

    def add_in_meta_file(self, filename):
        self.filewrite.write(filename)


    def generate_corrupt_images(self,count,parts=1):
        self.filewrite = open('images/kak_ygodno.txt', 'w')
        i=0
        if not os.path.exists(self.dir_x_images):
             os.makedirs(self.dir_x_images)
        if not os.path.exists(self.dir_new_y_images):
             os.makedirs(self.dir_new_y_images)
        for file_y in os.listdir(self.dir_y_images):
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
            #print(path_y_image)
            for j in range(count): 
                image=self.corrupt(np.copy(img_y),parts=parts) 
                file_x = imageName.replace(ext, "_"+str(j) + ext)
                path_x_image=os.path.join(self.dir_x_images,file_x)
                cv2.imwrite(path_x_image, image)
                self.add_in_meta_file(" " + file_x)
            self.filewrite.write("\n\r")
            i=i+1
        self.filewrite.close()

if __name__ == "__main__":
	myclass=Corrupter()
	myclass.generate_corrupt_images(30,parts=4)
#img = cv2.imread("photo/mix/4.jpg")
#cv2.imwrite("brigh3.jpg",myclass.increase_brightness(img,value=50,parts=5))




#cv2.imwrite("brigh3.jpg",increase_brightness(img,value=50))
#cv2.imwrite("brigh3.jpg",decrease_brightness(img,value=50))
#cv2.imwrite("contrast2.jpg",change_contrast(img,clipLimit=5.,titleGridSize=(12,12)))
#cv2.imwrite("gauss3.jpg",noisy_gauss(img,value=1000))
#cv2.imwrite("blur_image2.jpg",blur_image(img,blur_size=(6,6)))   
