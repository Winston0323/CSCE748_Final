import numpy as np
import cv2
import glob
import imutils
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools import *
import sys
if __name__ == "__main__": 

    Manual = False
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        print(arg)
        if arg == "0":
            print("Perform automatic stitching")
            Manual = False
        elif arg == "1":
            print("Perform Manual stitching")
            Manual = True
        else:
            print("Invalid argument, perform automatic stitching anyway")
    else:
        Manual = True
        
    image_path = glob.glob("Input/*.jpg")
    image_list = []
    # read all images
    for image in image_path:
        image_list.append(cv2.imread(image))
        cv2.imshow
    main_image = image_list[0]
    padded_main = padImage(main_image)
    pnt_main = corner_detect(main_image,image_path[0][6:-4])
    # loop through all the attaching image
    for i in range(1, len(image_list)):
        attach_img = image_list[i]
        height, width, channel = padded_main.shape
        pnt_attach = []
        pnt_main = []
        H = np.zeros((3, 3)) 
        # check if manual input
        if(Manual):
            pnt_attach , pnt_main = get_points(attach_img, padded_main)
            H, _ = cv2.findHomography(pnt_attach, pnt_main)
        else:
            pnt_attach, pnt_main = find_point_auto(attach_img, padded_main)
            H, _ = cv2.findHomography(pnt_attach, pnt_main, cv2.RANSAC, 2.0)
        #attach perspective image on to padded main image
        padded_main = warpPerspective(attach_img, padded_main, H, (height, width, channel))
        if Manual:
            cv2.imwrite('manual.jpg', padded_main)
        else:
            cv2.imwrite('auto.jpg', padded_main)

       
                
            
        