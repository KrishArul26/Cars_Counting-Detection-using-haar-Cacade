# This is a sample Python script.
import cv2

car_cascade_src = 'Required Files/cars.xml'
video_src = 'Required Files/cars.mp4'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(car_cascade_src)
video = cv2.VideoWriter('Required Files/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (450, 250))

while True:
    ret, img = cap.read()
    frame = img.copy()

    # Draw reference lines for counting
    cv2.line(img, (17, 646), (1363, 612), (0, 0, 255), 2)  # RED LINE
    # cv2.line(img, (220, 440), (1000, 440), (0, 255, 0), 1)  # Green
    # cv2.line(img, (220, 460), (1000, 460), (255, 0, 0), 1)

    if type(None) != type(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cars = car_cascade.detectMultiScale(gray, 1.1, 1)

        # for car in cars:
        car_count = 0
        for (x, y, w, h) in cars:
            # if 250 < x < 900 and y > 250:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(img, (x, y), (x + w, y), (51, 51, 255), -2)
            car_count += 1
            cv2.putText(img, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

            # cv2.imwrite(car)

        img = cv2.resize(img, (1200, 960))
        cv2.imshow('video', img)
        video.write(img)

        if cv2.waitKey(33) == 27:
            break
        continue

    break

cv2.destroyAllWindows()
