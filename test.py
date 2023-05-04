import cv2
import numpy as np
import matplotlib.pyplot as plt

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