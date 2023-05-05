import numpy as np
import cv2
import glob
import imutils
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_points(img_attach, img_main):
    print('Please select 4 points in each image for alignment.')
    plt.imshow(img_attach)
    p1, p2, p3, p4 = plt.ginput(4)
    plt.close()
    plt.imshow(img_main)
    p5, p6, p7, p8 = plt.ginput(4)
    plt.close()
    return np.array([p1, p2, p3, p4]) , np.array([p5, p6, p7, p8])

def padImage(image):
    
    height, width, channel = image.shape
    desired_height = height * 2
    desired_width = width * 2
    h_padding = height
    w_padding = width

    # Compute padding zone
    top = h_padding // 2
    bottom = h_padding - top
    left = w_padding // 2
    right = w_padding - left

    padded_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    resized_img = cv2.resize(padded_img, (desired_width, desired_height))
    return resized_img

def find_point_auto(img_attach, img_main):
    # Initialize detector
    detector = cv2.ORB_create()

    # Detect keypoints
    keypoints1, descriptors1 = detector.detectAndCompute(img_attach, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img_main, None)

    # Match the points using brute force
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    src_pts = []
    dst_pts = []
    for match in matches:
        src_keypoint = keypoints1[match.queryIdx]
        dst_keypoint = keypoints2[match.trainIdx]
       
        src_pts.append(np.array(src_keypoint.pt).reshape(1, 2))
        dst_pts.append(np.array(dst_keypoint.pt).reshape(1, 2))
    # convert to float 32 for findHomography
    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)
    return src_pts, dst_pts

def warpPerspective(img, main_image, H, output_shape):
    H_inv = np.linalg.inv(H)
    # loop through all pixel inside attaching image
    for i in tqdm(range(output_shape[0])):
        for j in range(output_shape[1]):
            coords = np.dot(H_inv, np.array([j, i, 1]))
            x = int(coords[0]/coords[2]) 
            y = int(coords[1]/coords[2])
            # Check if the x, y coordinates are within the bounds
            if x >= 0 and x < img.shape[1] and y >= 0 and y < img.shape[0]:
                # blend image
                if main_image[i, j, 0] +main_image[i, j, 1]+ main_image[i, j, 2] != 0.0:
                    main_image[i, j] = img[y, x]* 0.5 + main_image[i, j] * 0.5
                else:
                    main_image[i, j] = img[y, x]
    return main_image

def corner_detect(image, image_name):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Threshold the corner response
    threshold = 0.01 * corners.max()
    corners_thresh = np.where(corners > threshold)

    #local maxima
    win_size = 5
    max_corners = []
    for y, x in zip(corners_thresh[0], corners_thresh[1]):
        # Check if the current pixel is the local maximum in its neighborhood
        window = corners[y - win_size:y + win_size + 1, x - win_size:x + win_size + 1]
        if window.size > 0 and corners[y, x] == window.max():
            max_corners.append((x, y))

    for pos in max_corners:
        cv2.circle(image, (pos[0], pos[1]), 2, (0, 0, 255), -1)
        
    cv2.imwrite('image_' + image_name +'_corner.jpg', image)
    return np.array(max_corners)