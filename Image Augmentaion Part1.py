# Name: Aisangam
# Url: http://www.aisangam.com/
#Blog:http://www.aisangam.com/blog/
#Company: Aisangam
#Location: India
#YouTube Link: https://www.youtube.com/channel/UC9x_PL-LPk3Wp5V85F4GLHQ
#Discription: https://youtu.be/PePk_YkMQn0


import cv2
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import numpy as np
import os



def crop_image(image,y1,y2,x1,x2):
    image=image[y1:y2,x1:x2]
    cv2.imwrite(Folder_name+"/"+str(img_name)+"-Crop-"+str(x1)+str(x2)+"*"+str(y1)+str(y2)+Extension, image)

def padding_image(image,topBorder,bottomBorder,leftBorder,rightBorder,color_of_border=[0,0,0]):
    image = cv2.copyMakeBorder(image,topBorder,bottomBorder,leftBorder,
        rightBorder,cv2.BORDER_CONSTANT,value=color_of_border)
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-padd-" + str(topBorder) + str(bottomBorder) + "*" + str(leftBorder) + str(rightBorder) + Extension, image)

def flip_image(image,dir):
    image = cv2.flip(image, dir)
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-flip-" + str(dir)+Extension, image)

def superpixel_image(image,segments):
    seg=segments

    def segment_colorfulness(image, mask):
        # split the image into its respective RGB components, then mask
        # each of the individual RGB channels so we can compute
        # statistics only for the masked region
        (B, G, R) = cv2.split(image.astype("float"))
        R = np.ma.masked_array(R, mask=mask)
        G = np.ma.masked_array(B, mask=mask)
        B = np.ma.masked_array(B, mask=mask)

        # compute rg = R - G
        rg = np.absolute(R - G)

        # compute yb = 0.5 * (R + G) - B
        yb = np.absolute(0.5 * (R + G) - B)

        # compute the mean and standard deviation of both `rg` and `yb`,
        # then combine them
        stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
        meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))

        # derive the "colorfulness" metric and return it
        return stdRoot + (0.3 * meanRoot)

    orig = cv2.imread(image)
    vis = np.zeros(orig.shape[:2], dtype="float")

    # load the image and apply SLIC superpixel segmentation to it via
    # scikit-image
    image = io.imread(image)
    segments = slic(img_as_float(image), n_segments=segments,
                    slic_zero=True)
    for v in np.unique(segments):
        # construct a mask for the segment so we can compute image
        # statistics for *only* the masked region
        mask = np.ones(image.shape[:2])
        mask[segments == v] = 0

        # compute the superpixel colorfulness, then update the
        # visualization array
        C = segment_colorfulness(orig, mask)
        vis[segments == v] = C
    # scale the visualization image from an unrestricted floating point
    # to unsigned 8-bit integer array so we can use it with OpenCV and
    # display it to our screen
    vis = rescale_intensity(vis, out_range=(0, 255)).astype("uint8")

    # overlay the superpixel colorfulness visualization on the original
    # image
    alpha = 0.6
    overlay = np.dstack([vis] * 3)
    output = orig.copy()
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    # cv2.imshow("Visualization", vis)
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-superpixels-" + str(seg) + Extension, output)

def invert_image(image,channel):
    # image=cv2.bitwise_not(image)
    image=(channel-image)
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-invert-"+str(channel)+Extension, image)

def add_light(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    image=cv2.LUT(image, table)
    if gamma>=1:
        cv2.imwrite(Folder_name +"/"+str(img_name)+ "-light-"+str(gamma)+Extension, image)
    else:
        cv2.imwrite(Folder_name +"/"+str(img_name)+ "-dark-" + str(gamma) + Extension, image)

def add_light_color(image, color, gamma=1.0):
    invGamma = 1.0 / gamma
    image = (color - image)
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    image=cv2.LUT(image, table)
    if gamma>=1:
        cv2.imwrite(Folder_name +"/"+str(img_name)+ "-light_color-"+str(gamma)+Extension, image)
    else:
        cv2.imwrite(Folder_name +"/"+str(img_name)+ "-dark_color" + str(gamma) + Extension, image)

def saturation_image(image,saturation):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    v = image[:, :, 2]
    v = np.where(v <= 255 - saturation, v + saturation, 255)
    image[:, :, 2] = v

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-saturation-" + str(saturation) + Extension, image)

def hue_image(image,saturation):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    v = image[:, :, 2]
    v = np.where(v <= 255 + saturation, v - saturation, 255)
    image[:, :, 2] = v

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-hue-" + str(saturation) + Extension, image)

def multiply_image(image,R,G,B):
    image=image*[R,G,B]
    cv2.imwrite(Folder_name+"/"+str(img_name)+"-Multiply-"+str(R)+"*"+str(G)+"*"+str(B)+Extension, image)

def gausian_blur(image,blur):
    image = cv2.GaussianBlur(image,(5,5),blur)
    cv2.imwrite(Folder_name+"/"+str(img_name)+"-GausianBLur-"+str(blur)+Extension, image)

def averageing_blur(image,shift):
    image=cv2.blur(image,(shift,shift))
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-AverageingBLur-" + str(shift) + Extension, image)

def median_blur(image,shift):
    image=cv2.medianBlur(image,shift)
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-MedianBLur-" + str(shift) + Extension, image)

def bileteralBlur(image,d,color,space):
    image = cv2.bilateralFilter(image, d,color,space)
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-BileteralBlur-"+str(d)+"*"+str(color)+"*"+str(space)+ Extension, image)

def erosion_image(image,shift):
    kernel = np.ones((shift,shift),np.uint8)
    image = cv2.erode(image,kernel,iterations = 1)
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Erosion-"+"*"+str(shift) + Extension, image)

def dilation_image(image,shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.dilate(image,kernel,iterations = 1)
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Dilation-" + "*" + str(shift)+ Extension, image)

def opening_image(image,shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Opening-" + "*" + str(shift)+ Extension, image)

def closing_image(image, shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Closing-" + "*" + str(shift) + Extension, image)

def morphological_gradient_image(image, shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Morphological_Gradient-" + "*" + str(shift) + Extension, image)

def top_hat_image(image, shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Top_Hat-" + "*" + str(shift) + Extension, image)

def black_hat_image(image, shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Black_Hat-" + "*" + str(shift) + Extension, image)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    cv2.imwrite(Folder_name+"/"+str(img_name)+"-Sharpen-"+Extension, image)

def emboss_image(image):
    kernel_emboss_1=np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
    image = cv2.filter2D(image, -1, kernel_emboss_1)+128
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Emboss-" + Extension, image)

def edge_image(image,ksize):
    image = cv2.Sobel(image,cv2.CV_16U,1,0,ksize=ksize)
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Edge-"+str(ksize) + Extension, image)

def addeptive_gaussian_noise(image):
    h,s,v=cv2.split(image)
    s = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    h = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    v = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    image=cv2.merge([h,s,v])
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Addeptive_gaussian_noise-" + Extension, image)

def salt_image(image,p,a):
    noisy=image
    num_salt = np.ceil(a * image.size * p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[coords] = 1
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Salt-"+str(p)+"*"+str(a) + Extension, image)

def paper_image(image,p,a):
    noisy=image
    num_pepper = np.ceil(a * image.size * (1. - p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    noisy[coords] = 0
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Paper-" + str(p) + "*" + str(a) + Extension, image)

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
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Salt_And_Paper-" + str(p) + "*" + str(a) + Extension, image)

def contrast_image(image,contrast):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[:,:,2] = [[max(pixel - contrast, 0) if pixel < 190 else min(pixel + contrast, 255) for pixel in row] for row in image[:,:,2]]
    image= cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Contrast-" + str(contrast) + Extension, image)

def edge_detect_canny_image(image,th1,th2):
    image = cv2.Canny(image,th1,th2)
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Edge Canny-" + str(th1) + "*" + str(th2) + Extension, image)

def grayscale_image(image):
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Grayscale-" + Extension, image)

def scale_image(image,fx,fy):
    image = cv2.resize(image,None,fx=fx, fy=fy, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(Folder_name+"/"+str(img_name)+"-Scale-"+str(fx)+str(fy)+Extension, image)

def translation_image(image,x,y):
    rows, cols ,c= image.shape
    M = np.float32([[1, 0, x], [0, 1, y]])
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Translation-" + str(x) + str(y) + Extension, image)

def rotate_image(image,deg):
    rows, cols,c = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), deg, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Rotate-" + str(deg) + Extension, image)

def transformation_image(image):
    rows, cols, ch = image.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Transformations-" + str(1) + Extension, image)

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[100, 10], [200, 50], [0, 150]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Transformations-" + str(2) + Extension, image)

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[100, 10], [200, 50], [30, 175]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Transformations-" + str(3) + Extension, image)

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[100, 10], [200, 50], [70, 150]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name +"/"+str(img_name)+ "-Transformations-" + str(4) + Extension, image)


# Folder_name="combined"
Extension=".jpg"
dataset_dir=''

image="test.jpg"
img_name=image.split(".")[0]
if not os.path.exists(img_name):
    os.makedirs(img_name)
else:
    print ("Problem in creating directory, Already directory exist")

Folder_name=os.path.join(str(dataset_dir),(img_name))

w=450
h=400

image=cv2.imread(image)

image=cv2.resize(image,(w,h))
image_file=Folder_name+"/"+str(img_name)+"-Resize-"+str(w)+"*"+str(h)+Extension
cv2.imwrite(image_file, image)

print (image_file)
image=cv2.imread(image_file)



crop_image(image,100,400,0,350)#(y1,y2,x1,x2)(bottom,top,left,right)
crop_image(image,100,400,100,450)#(y1,y2,x1,x2)(bottom,top,left,right)
crop_image(image,0,300,0,350)#(y1,y2,x1,x2)(bottom,top,left,right)
crop_image(image,0,300,100,450)#(y1,y2,x1,x2)(bottom,top,left,right)
crop_image(image,100,300,100,350)#(y1,y2,x1,x2)(bottom,top,left,right)

padding_image(image,100,0,0,0)#(y1,y2,x1,x2)(bottom,top,left,right)
padding_image(image,0,100,0,0)#(y1,y2,x1,x2)(bottom,top,left,right)
padding_image(image,0,0,100,0)#(y1,y2,x1,x2)(bottom,top,left,right)
padding_image(image,0,0,0,100)#(y1,y2,x1,x2)(bottom,top,left,right)
padding_image(image,100,100,100,100)#(y1,y2,x1,x2)(bottom,top,left,right)

flip_image(image,0)#horizontal
flip_image(image,1)#vertical
flip_image(image,-1)#both

superpixel_image(image_file,100)
superpixel_image(image_file,50)
superpixel_image(image_file,25)
superpixel_image(image_file,75)
superpixel_image(image_file,200)

invert_image(image,255)
invert_image(image,200)
invert_image(image,150)
invert_image(image,100)
invert_image(image,50)

add_light(image,1.5)
add_light(image,2.0)
add_light(image,2.5)
add_light(image,3.0)
add_light(image,4.0)
add_light(image,5.0)
add_light(image,0.7)
add_light(image,0.4)
add_light(image,0.3)
add_light(image,0.1)

add_light_color(image,255,1.5)
add_light_color(image,200,2.0)
add_light_color(image,150,2.5)
add_light_color(image,100,3.0)
add_light_color(image,50,4.0)
add_light_color(image,255,0.7)
add_light_color(image,150,0.3)
add_light_color(image,100,0.1)

saturation_image(image,50)
saturation_image(image,100)
saturation_image(image,150)
saturation_image(image,200)

hue_image(image,50)
hue_image(image,100)
hue_image(image,150)
hue_image(image,200)



multiply_image(image,0.5,1,1)
multiply_image(image,1,0.5,1)
multiply_image(image,1,1,0.5)
multiply_image(image,0.5,0.5,0.5)

multiply_image(image,0.25,1,1)
multiply_image(image,1,0.25,1)
multiply_image(image,1,1,0.25)
multiply_image(image,0.25,0.25,0.25)

multiply_image(image,1.25,1,1)
multiply_image(image,1,1.25,1)
multiply_image(image,1,1,1.25)
multiply_image(image,1.25,1.25,1.25)

multiply_image(image,1.5,1,1)
multiply_image(image,1,1.5,1)
multiply_image(image,1,1,1.5)
multiply_image(image,1.5,1.5,1.5)


gausian_blur(image,0.25)
gausian_blur(image,0.50)
gausian_blur(image,1)
gausian_blur(image,2)
gausian_blur(image,4)

averageing_blur(image,5)
averageing_blur(image,4)
averageing_blur(image,6)

median_blur(image,3)
median_blur(image,5)
median_blur(image,7)

bileteralBlur(image,9,75,75)
bileteralBlur(image,12,100,100)
bileteralBlur(image,25,100,100)
bileteralBlur(image,40,75,75)

erosion_image(image,1)
erosion_image(image,3)
erosion_image(image,6)

dilation_image(image,1)
dilation_image(image,3)
dilation_image(image,5)


opening_image(image,1)
opening_image(image,3)
opening_image(image,5)

closing_image(image,1)
closing_image(image,3)
closing_image(image,5)

morphological_gradient_image(image,5)
morphological_gradient_image(image,10)
morphological_gradient_image(image,15)

top_hat_image(image,200)
top_hat_image(image,300)
top_hat_image(image,500)

black_hat_image(image,200)
black_hat_image(image,300)
black_hat_image(image,500)


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


scale_image(image,0.3,0.3)
scale_image(image,0.7,0.7)
scale_image(image,2,2)
scale_image(image,3,3)

translation_image(image,150,150)
translation_image(image,-150,150)
translation_image(image,150,-150)
translation_image(image,-150,-150)

rotate_image(image,90)
rotate_image(image,180)
rotate_image(image,270)

transformation_image(image)




