import numpy as np
import cv2
import glob
import imutils
import matplotlib.pyplot as plt


    
def get_points(img_attach, img_main):
    print('Please select 4 points in each image for alignment.')
    plt.imshow(img_attach)
    p1, p2, p3, p4 = plt.ginput(4)
    plt.close()
    plt.imshow(img_main)
    p5, p6, p7, p8 = plt.ginput(4)
    plt.close()
    return (p1, p2, p3, p4, p5, p6, p7, p8)
if __name__ == "__main__": 
    image_path = glob.glob("Input/*.jpg")
    image_list = []

    for image in image_path:
        image_list.append(cv2.imread(image))
        cv2.imshow
    get_points(image_list[0], image_list[1])
        