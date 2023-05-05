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

    padded_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    resized_img = cv2.resize(padded_img, (desired_width, desired_height))
    return resized_img
def find_point_auto(img_attach, img_main):
    # Initialize the feature detector and descriptor extractor
    detector = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = detector.detectAndCompute(img_attach, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img_main, None)

    # Match the descriptors using the BFMatcher
    matches = matcher.match(descriptors1, descriptors2)

    # Use RANSAC to estimate the homography matrix
    src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    return src_pts, dst_pts

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
                
                if main_image[i, j, 0] +main_image[i, j, 1]+ main_image[i, j, 2] != 0.0:
                    main_image[i, j] = img[y, x]* 0.5 + main_image[i, j] * 0.5
                else:
                    main_image[i, j] = img[y, x]
    return main_image

def corner_detect(image, image_name):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set Harris detector parameters
    block_size = 3
    ksize = 3
    k = 0.04

    # Apply Harris corner detection algorithm
    corners = cv2.cornerHarris(gray, block_size, ksize, k)

    # Threshold the corner response to retain only the strongest corners
    threshold = 0.01 * corners.max()
    corners_thresh = np.where(corners > threshold)

    # Find the local maxima of the thresholded corner response
    win_size = 5
    max_corners = []
    for y, x in zip(corners_thresh[0], corners_thresh[1]):
        # Check if the current pixel is the local maximum in its neighborhood
        window = corners[y - win_size:y + win_size + 1, x - win_size:x + win_size + 1]
        if window.size > 0 and corners[y, x] == window.max():
            max_corners.append((x, y))

    for pos in max_corners:
        cv2.circle(image, (pos[0], pos[1]), 3, (0, 0, 255), -1)
    print('image_' + image_name +'_corner.jpg')
    cv2.imwrite('image_' + image_name +'_corner.jpg', image)
    return np.array(max_corners)