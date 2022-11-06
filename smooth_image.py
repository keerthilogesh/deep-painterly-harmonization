import numpy as np
import cv2 as cv

# LPF helps in removing noise, blurring edges in the images
# Averaging
# Gaussian blurring - Highly effective in removing gaussian noise from the image
# Median blurring - Highly effective against salt and pepper noise
# HPF filters help in finding edges in images

from matplotlib import pyplot as plt
img = cv.imread("C:/Users/keert/PycharmProjects/deep-painterly-harmonization/output/image_implanting/1_naive.jpg")
n=11
kernel = np.ones((n,n),np.float32)/(n**2)
dst = cv.filter2D(img,-1,kernel)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()

# smooth image
# blend image
# style transfer