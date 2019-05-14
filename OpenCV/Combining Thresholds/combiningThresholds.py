# import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # if orient == 'x':
    #     sobel = cv2.Sobel( img, cv2.CV_64F, 1, 0, ksize = sobel_kernel )
    # elif orient == 'y':
    #     sobel = cv2.Sobel( img, cv2.CV_64F, 1, 0, ksize = sobel_kernel )

    # abs_sobel = np.absolute(sobel)
    # scaled_sobel = np.uint8( abs_sobel * 255 / np.max(abs_sobel) )

    # grad_binary = np.zeros_like(img)
    # grad_binary [ ( thresh[0] <= scaled_sobel ) & ( scaled_sobel <= thresh[1]) ] = 1
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    else:
        raise Exception('Orientation should be x or y')
    
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8( abs_sobel * 255 / np.max(abs_sobel) )
    
    sxBinary = np.zeros_like(scaled_sobel)
    sxBinary[ (thresh[0] < scaled_sobel) & (scaled_sobel < thresh[1]) ] = 1
      
    binary_output = sxBinary # np.copy(img) # Remove this line
    return binary_output
    # return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    return dir_binary


image = mpimg.imread('Combining Thresholds/signs_vehicles_xygrad.png')
cv2.imshow('image', image)

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements
# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(0, 255))
# mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(0, 255))
# dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))


# image = mpimg.imread('Combining Thresholds/signs_vehicles_xygrad.png')
# plt.imshow(image)

# cv2.waitKey(0)
# cv2.imshow('image', gradx, cmap='gray')
# cv2.waitKey(0)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(gradx, cmap='gray')
ax2.set_title('Thresholded Gradient', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

plt.show()

cv2.waitKey(0)