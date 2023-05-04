import numpy as np
import cv2
import glob
import imutils
import matplotlib.pyplot as plt
from tqdm import tqdm
def warpPerspective(img, main_image, H, output_shape):

    H_inv = np.linalg.inv(H)
    for i in tqdm(range(output_shape[0])):
        for j in range(output_shape[1]):
            # Calculate the corresponding (x, y) coordinates in the input image
            coords = np.dot(H_inv, np.array([j, i, 1]))
            x = int(coords[0]/coords[2]) 
            y = int(coords[1]/coords[2])
            # Check if the (x, y) coordinates are within the bounds of the input image
            if x >= 0 and x < img.shape[1] and y >= 0 and y < img.shape[0]:
                # If the (x, y) coordinates are valid, set the corresponding pixel in the output image
                main_image[i, j] = img[y, x]
    
    return main_image

def padImage(image):

    # Get the current size of the image
    height, width, channel = image.shape
    desired_height = height * 2
    desired_width = width * 2
    # Compute the amount of padding needed
    h_padding = max(0, desired_height - height)
    w_padding = max(0, desired_width - width)

    # Compute the top, bottom, left, and right padding sizes
    top = h_padding // 2
    bottom = h_padding - top
    left = w_padding // 2
    right = w_padding - left

    # Pad the image with black pixels
    padded_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Resize the padded image to the desired size
    resized_img = cv2.resize(padded_img, (desired_width, desired_height))
    return resized_img
    
def get_points(img_attach, img_main):
    print('Please select 4 points in each image for alignment.')
    plt.imshow(img_attach)
    p1, p2, p3, p4 = plt.ginput(4)
    plt.close()
    plt.imshow(img_main)
    p5, p6, p7, p8 = plt.ginput(4)
    plt.close()
    return np.array([p1, p2, p3, p4]) , np.array([p5, p6, p7, p8])
if __name__ == "__main__": 
    image_path = glob.glob("Input/*.jpg")
    image_list = []

    for image in image_path:
        image_list.append(cv2.imread(image))
        cv2.imshow
    main_image = image_list[0]
    padded_main = padImage(main_image)
    # loop through all the attaching image
    for i in range(1, len(image_list)):
        attach_img = image_list[i]
        height, width, channel = padded_main.shape
        pnt_attach , pnt_main = get_points(attach_img, padded_main)
        H, _ = cv2.findHomography(pnt_attach, pnt_main)
        H_inv = np.linalg.inv(H)
        result = np.zeros((height, width, channel))

        attach_hgt, attach_wdt, channel = attach_img.shape

        padded_main = warpPerspective(attach_img, padded_main, H, (height, width, channel))
        #cv2.imshow("main",padded_main)
        cv2.imwrite('output.jpg', padded_main)

       
                
            
        