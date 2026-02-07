import cv2
import numpy as np

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

img = cv2.imread("/Users/vishalpande/Projects/GenAI-Machine-Learning-Deep-Learning/Day12-computer-vision-object-detector/data/Images/5EDF996C-3A02-4A5D-BE42-C2F626AFFC75_1_105_c.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=1)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray, 1.09)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(
            roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
        )

cv2.imshow("Face & Eye Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
