import cv2
from PIL import Image
import numpy as np

def pre_processing_image(fig):
    image = fig.resize((450, 250))
    image_arr = np.array(image)
    grey = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
    Image.fromarray(grey)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    Image.fromarray(blur)
    dilated = cv2.dilate(blur, np.ones((3, 3)))
    Image.fromarray(dilated)
    return image_arr, grey
