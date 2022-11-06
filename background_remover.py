import cv2
import numpy as np
from utils import imshow


def background_remover(image=None):
    # img = r"C:\Users\keert\PycharmProjects\deep-painterly-harmonization\ip_data\background_image\dog_sample.jpg"
    # image = cv2.imread(img)
    copy = image.copy()
    edges = cv2.Canny(image, 80, 150)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    erosion = cv2.morphologyEx(closing, cv2.MORPH_ERODE, kernel, iterations=1)

    # When using Grabcut the mask image should be:
    #    0 - sure background
    #    1 - sure foreground
    #    2 - unknown

    mask = np.zeros(image.shape[:2], np.uint8)
    mask[:] = 2
    mask[erosion == 255] = 1
    # Create a mask (of zeros uint8 datatype) that is the same size (width, height) as our original image
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    out_mask = mask.copy()
    x, y, w, h = cv2.selectROI("select the area", image)
    start = (x, y)
    end = (x + w, y + h)
    rect = (x, y, w, h)

    cv2.rectangle(copy, start, end, (0, 0, 255), 3)
    imshow("Input Image", copy)

    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 100, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image = image * mask2[:, :, np.newaxis]
    mask2_invert = cv2.bitwise_not(mask2 * 255)
    # imshow("Mask", mask * 80)
    # imshow("Mask2", mask2 * 255)
    # imshow("Image", image)
    # imshow("", image)
    # imshow("d", mask2_invert)
    # imshow("df", mask2 * 255)
    # cv2.imwrite(img.split(".jpg")[0] + "_bg.jpg", image)
    # cv2.imwrite(img.split(".jpg")[0] + "_mask.jpg", mask2 * 255)
    # cv2.imwrite(img.split(".jpg")[0] + "_maskinvert.jpg", mask2_invert)
    # cv2.imwrite(img.split(".jpg")[0] + "_bg.jpg", image)
    imshow("background removed image", image)
    imshow("background removed image mask", mask2*255)
    return image, mask2*255
