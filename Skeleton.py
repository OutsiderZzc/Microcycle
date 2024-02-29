import numpy as np
from skimage import io, morphology, img_as_bool
from matplotlib import pyplot as plt
import cv2
def skeleton(bin_img):
    sk_img = morphology.skeletonize(bin_img)
    plt.imshow(sk_img, cmap="gray")
    plt.axis('off')
    plt.show()

    res = (sk_img * 255).astype(np.uint8)
    return res

def morph_find(binary):
    img = binary.copy()
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    finished = False
    size = np.size(binary)
    skeleton = np.zeros(binary.shape, np.uint8)
    while (not finished):
        eroded = cv2.erode(binary, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(binary, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary = eroded.copy()

        zeros = size - cv2.countNonZero(binary)
        if zeros == size:
            finished = True

    contours, hireachy = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plt.imshow(skeleton)
    plt.axis('off')
    plt.show()