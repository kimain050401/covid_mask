import numpy as np
import cv2
import dlib
import datetime
import time

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 각 점 위치에 따른 얼굴 부위 구분
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

num = 0

def detect(gray, frame):
    global num
    # 얼굴 찾기
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

    # 얼굴 주요 부분들 찾기
    for (x, y, w, h) in faces:
        num = 1
        # openCV 이미지를 dlib 이미지로 변경
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        # 주요 부분 포인트 지정
        landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, dlib_rect).parts()])
        # 원하는 포인트 지정
        landmarks_display = landmarks[48:61]

        # 포인트 출력
        for idx, point in enumerate(landmarks_display):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(frame, pos, 2, color=(0, 255, 255), thickness=-1)

    return frame

# 웹캠에서 이미지 가져오기
video_capture = cv2.VideoCapture(0)

while True:
    # 웹캠 이미지를 프레임으로 자름
    x, frame = video_capture.read()
    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 만들어준 얼굴 눈 찾기
    canvas = detect(gray, frame)
    # 찾은 이미지 보여주기
    cv2.imshow("haha", canvas)
    
    # q 누르면 종료
    if num == 1:
        now = datetime.datetime.now()
        timegood = now.strftime("%Y_%m_%d %H_%M_%S")
        print(timegood)
        name = "C:/covid_mask/" + timegood + ".jpg"
        cv2.imwrite(name, canvas)
        num = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
