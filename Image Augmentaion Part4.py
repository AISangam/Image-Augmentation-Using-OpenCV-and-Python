#Name: Aisangam
#Url: http://www.aisangam.com/
#Blog:http://www.aisangam.com/blog/
#Company: Aisangam
#YouTube Link: https://www.youtube.com/channel/UC9x_PL-LPk3Wp5V85F4GLHQ
#Discription: https://youtu.be/AVr_DjnAlNw?list=PLCK5Mm9zwPkFt1iX30kD5eJ9hy-EeijQn

import cv2
import numpy as np

Folder_name="augmented_image_part4"
Extension=".jpg"

def scale_image(image,fx,fy):
    image = cv2.resize(image,None,fx=fx, fy=fy, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(Folder_name+"/Scale-"+str(fx)+str(fy)+Extension, image)

def translation_image(image,x,y):
    rows, cols ,c= image.shape
    M = np.float32([[1, 0, x], [0, 1, y]])
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/Translation-" + str(x) + str(y) + Extension, image)

def rotate_image(image,deg):
    rows, cols,c = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), deg, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/Rotate-" + str(deg) + Extension, image)

def transformation_image(image):
    rows, cols, ch = image.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/Transformations-" + str(1) + Extension, image)

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[100, 10], [200, 50], [0, 150]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/Transformations-" + str(2) + Extension, image)

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[100, 10], [200, 50], [30, 175]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/Transformations-" + str(3) + Extension, image)

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[100, 10], [200, 50], [70, 150]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/Transformations-" + str(4) + Extension, image)

image_file="Resize-450*400.jpg"
image=cv2.imread(image_file)

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

