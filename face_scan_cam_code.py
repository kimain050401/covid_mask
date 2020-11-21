# 작업하려는 컴퓨터에서 뒤 링크 접속 후 데이터셋 다운로드 https://diy-project.tistory.com/attachment/cfile21.uf@99B5B24E5B5C84B510E365.xml

import cv2
import time

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1)

while (True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        now = time.strftime('%Y_%m_%d-%H_%M_%S')
        now = "covid_mask/" + now + ".jpg"
        cv2.imwrite(now, img)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
