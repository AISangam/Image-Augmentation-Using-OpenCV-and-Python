#Name: Aisangam
#Url: http://www.aisangam.com/
#Blog: http://www.aisangam.com/blog/
#Company: Aisangam
#YouTube Link: https://www.youtube.com/channel/UC9x_PL-LPk3Wp5V85F4GLHQ
#Discription: https://youtu.be/NQUsbkZsCjc?list=PLCK5Mm9zwPkFt1iX30kD5eJ9hy-EeijQn

import cv2
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import numpy as np
# (x1, y1) (left, top)
# (right, bottom) (x2, y2)

# (top,right,bottom,left)
# (32,64,0,0)

Folder_name="augmented_image_part3"
Extension=".jpg"

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    cv2.imwrite(Folder_name+"/Sharpen-"+Extension, image)

def emboss_image(image):
    kernel_emboss_1=np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
    image = cv2.filter2D(image, -1, kernel_emboss_1)+128
    cv2.imwrite(Folder_name + "/Emboss-" + Extension, image)

def edge_image(image,ksize):
    image = cv2.Sobel(image,cv2.CV_16U,1,0,ksize=ksize)
    cv2.imwrite(Folder_name + "/Edge-"+str(ksize) + Extension, image)

def addeptive_gaussian_noise(image):
    h,s,v=cv2.split(image)
    s = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    h = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    v = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    image=cv2.merge([h,s,v])
    cv2.imwrite(Folder_name + "/Addeptive_gaussian_noise-" + Extension, image)

def salt_image(image,p,a):
    noisy=image
    num_salt = np.ceil(a * image.size * p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[coords] = 1
    cv2.imwrite(Folder_name + "/Salt-"+str(p)+"*"+str(a) + Extension, image)

def paper_image(image,p,a):
    noisy=image
    num_pepper = np.ceil(a * image.size * (1. - p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    noisy[coords] = 0
    cv2.imwrite(Folder_name + "/Paper-" + str(p) + "*" + str(a) + Extension, image)

def salt_and_paper_image(image,p,a):
    noisy=image
    #salt
    num_salt = np.ceil(a * image.size * p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[coords] = 1

    #paper
    num_pepper = np.ceil(a * image.size * (1. - p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    noisy[coords] = 0
    cv2.imwrite(Folder_name + "/Salt_And_Paper-" + str(p) + "*" + str(a) + Extension, image)

def contrast_image(image,contrast):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[:,:,2] = [[max(pixel - contrast, 0) if pixel < 190 else min(pixel + contrast, 255) for pixel in row] for row in image[:,:,2]]
    image= cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(Folder_name + "/Contrast-" + str(contrast) + Extension, image)

def edge_detect_canny_image(image,th1,th2):
    image = cv2.Canny(image,th1,th2)
    cv2.imwrite(Folder_name + "/Edge Canny-" + str(th1) + "*" + str(th2) + Extension, image)

def grayscale_image(image):
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(Folder_name + "/Grayscale-" + Extension, image)


image_file="Resize-450*400.jpg"
image=cv2.imread(image_file)


sharpen_image(image)
emboss_image(image)

edge_image(image,1)
edge_image(image,3)
edge_image(image,5)
edge_image(image,9)

addeptive_gaussian_noise(image)

salt_image(image,0.5,0.009)
salt_image(image,0.5,0.09)
salt_image(image,0.5,0.9)


paper_image(image,0.5,0.009)
paper_image(image,0.5,0.09)
paper_image(image,0.5,0.9)

salt_and_paper_image(image,0.5,0.009)
salt_and_paper_image(image,0.5,0.09)
salt_and_paper_image(image,0.5,0.9)

contrast_image(image,25)
contrast_image(image,50)
contrast_image(image,100)

edge_detect_canny_image(image,100,200)
edge_detect_canny_image(image,200,400)

grayscale_image(image)


