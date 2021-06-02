# This is a script for the counting the buses and cars in the single frame using OpenCV
import cv2
from PIL import Image
from image_pre_processing import pre_processing_image

# Import cascade model

car_cascade_src = 'Required Files/cars.xml'
bus_cascade_src = 'Required Files/Bus_front.xml'

# Load the image
fig = 'Required Files/car1.jpg'

# Read the image
image = Image.open(fig)
# Preprocessing for images
image_arr, grey = pre_processing_image(image)

# create the Cascade Classifier
car_cascade = cv2.CascadeClassifier(car_cascade_src)
cars = car_cascade.detectMultiScale(grey, 1.1, 1)

# Counting the bus number if it presented
bus_count = 0
bus_cascade = cv2.CascadeClassifier(bus_cascade_src)
bus = bus_cascade.detectMultiScale(grey, 1.1, 1)
for (x, y, w, h) in bus:
    cv2.rectangle(image_arr, (x, y), (x + w, y + h), (0, 255, 0), 2)
    bus_count += 1
print(bus_count, "buses are counted")

# Counting the car number if it presented
car_count = 0
for (x, y, w, h) in cars:
    cv2.rectangle(image_arr, (x, y), (x + w, y + h), (255, 0, 0), 2)
    car_count += 1
print(car_count, "cars are counted")

# save the images after the counting as well as detection
# img = Image.fromarray(image_arr, 'RGB')
# path = 'D:/Python_deployment_project/vechicle_detection/test_count_jpg'
# cv2.imwrite(path, img)
