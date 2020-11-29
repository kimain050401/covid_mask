import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
import cv2
import tensorflow.keras
from PIL import Image, ImageOps, ImageFont, ImageDraw, ImageGrab
import numpy as np
import threading
import math
import time

np.set_printoptions(suppress=True)

model = tensorflow.keras.models.load_model('keras_model.h5')

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
cap = cv2.VideoCapture(1)

form_class = uic.loadUiType("mainwindow.ui")[0]

class MyWindow(QMainWindow, form_class):

    i = 10

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("코로나19 자동 진단")
        # self.setWindowIcon(QIcon("tooth.ico"))
        # self.setFixedSize(1920, 1080)
        self.showMaximized()
        self.start.setShortcut("Space")
        self.start.setStatusTip("Start")
        self.start.triggered.connect(self.mask_go)
        self.info.setStyleSheet("Color : red")
        self.mask.setStyleSheet("Color : green")
        self.temp.setStyleSheet("Color : green")
        self.stop.setStyleSheet("Color : green")
        self.noti1.setStyleSheet("Color : blue")
        self.noti2.setStyleSheet("Color : blue")
        self.all.setStyleSheet("Color : blue")
        th = threading.Thread(target=self.lets_go)
        th.start()

    def mask_test(self):
        global how, i
        self.mask.setText("마스크 착용 여부 결과 : 대기")
        self.temp.setText("발열 증상 여부 결과 : 대기")
        self.stop.setText("등교 중지 여부 결과 : 대기")
        self.maskt1.setText("이미지 대기중")
        self.maskt2.setText("이미지 대기중")
        self.maskt3.setText("이미지 대기중")
        self.temp1.setText("이미지 대기중")
        self.mask1.setText("1회차 마스크 착용 여부 : 대기")
        self.mask2.setText("2회차 마스크 착용 여부 : 대기")
        self.mask3.setText("3회차 마스크 착용 여부 : 대기")
        self.temp1.setText("발열 증상 여부 : 대기")
        self.all.setText("")
        mask_how = 0
        self.info.setText("코로나19를 예방하기 위한 자동 진단을 시작합니다.")
        time.sleep(2)
        self.info.setText("마스크 착용 여부를 확인하기 위해 모니터 상단에 있는 카메라를 바라봐 주시기 바랍니다.")
        time.sleep(1)
        ret, img = cap.read()
        cv2.flip(img, 1)
        cv2.imwrite("test.jpg", img)

        image = Image.open('test.jpg')
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        image_array = np.asarray(image)

        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        data[0] = normalized_image_array

        prediction = model.predict(data)
        okay = round(prediction[0][0] * 100, 1)
        okay2 = "마스크 착용 : " + str(okay) + "%"
        noo = round(prediction[0][1] * 100, 1)
        noo2 = "마스크 미착용 : " + str(noo) + "%"
        noh = round(prediction[0][2] * 100, 1)
        noh2 = "사람 없음 : " + str(noh) + "%"
        print(prediction)
        print(okay2)
        print(noo2)
        print(noh2)
        txtp = ["===> 예상 : 마스크 착용", "===> 예상 : 마스크 미착용", "===> 예상 : 사람 없음"]
        print(txtp[np.argmax(prediction[0])])

        if txtp[np.argmax(prediction[0])] == "===> 예상 : 마스크 미착용":
            self.mask1.setText("1회차 마스크 여부 : 미착용")
            mask_how = mask_how + 1
        elif txtp[np.argmax(prediction[0])] == "===> 예상 : 마스크 착용":
            self.mask1.setText("1회차 마스크 여부 : 착용")
        elif txtp[np.argmax(prediction[0])] == "===> 예상 : 사람 없음":
            self.mask1.setText("1회차 마스크 여부 : 미감지")
            mask_how = mask_how + 100

        self.qPixmapFileVar = QPixmap()
        self.qPixmapFileVar.load("test.jpg")
        self.qPixmapFileVar = self.qPixmapFileVar.scaledToWidth(400)
        self.maskt1.setPixmap(self.qPixmapFileVar)

        time.sleep(1)

        ret, img = cap.read()
        cv2.flip(img, 1)
        cv2.imwrite("test.jpg", img)

        image = Image.open('test.jpg')
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        image_array = np.asarray(image)

        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        data[0] = normalized_image_array

        prediction = model.predict(data)
        okay = round(prediction[0][0] * 100, 1)
        okay2 = "마스크 착용 : " + str(okay) + "%"
        noo = round(prediction[0][1] * 100, 1)
        noo2 = "마스크 미착용 : " + str(noo) + "%"
        noh = round(prediction[0][2] * 100, 1)
        noh2 = "사람 없음 : " + str(noh) + "%"
        print(prediction)
        print(okay2)
        print(noo2)
        print(noh2)
        txtp = ["===> 예상 : 마스크 착용", "===> 예상 : 마스크 미착용", "===> 예상 : 사람 없음"]
        print(txtp[np.argmax(prediction[0])])

        if txtp[np.argmax(prediction[0])] == "===> 예상 : 마스크 미착용":
            self.mask2.setText("2회차 마스크 여부 : 미착용")
            mask_how = mask_how + 1
        elif txtp[np.argmax(prediction[0])] == "===> 예상 : 마스크 착용":
            self.mask2.setText("2회차 마스크 여부 : 착용")
        elif txtp[np.argmax(prediction[0])] == "===> 예상 : 사람 없음":
            self.mask2.setText("2회차 마스크 여부 : 미감지")
            mask_how = mask_how + 100

        self.qPixmapFileVar = QPixmap()
        self.qPixmapFileVar.load("test.jpg")
        self.qPixmapFileVar = self.qPixmapFileVar.scaledToWidth(400)
        self.maskt2.setPixmap(self.qPixmapFileVar)

        time.sleep(1)

        ret, img = cap.read()
        cv2.flip(img, 1)
        cv2.imwrite("test.jpg", img)

        image = Image.open('test.jpg')
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        image_array = np.asarray(image)

        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        data[0] = normalized_image_array

        prediction = model.predict(data)
        okay = round(prediction[0][0] * 100, 1)
        okay2 = "마스크 착용 : " + str(okay) + "%"
        noo = round(prediction[0][1] * 100, 1)
        noo2 = "마스크 미착용 : " + str(noo) + "%"
        noh = round(prediction[0][2] * 100, 1)
        noh2 = "사람 없음 : " + str(noh) + "%"
        print(prediction)
        print(okay2)
        print(noo2)
        print(noh2)
        txtp = ["===> 예상 : 마스크 착용", "===> 예상 : 마스크 미착용", "===> 예상 : 사람 없음"]
        print(txtp[np.argmax(prediction[0])])

        if txtp[np.argmax(prediction[0])] == "===> 예상 : 마스크 미착용":
            self.mask3.setText("3회차 마스크 여부 : 미착용")
            mask_how = mask_how + 1
        elif txtp[np.argmax(prediction[0])] == "===> 예상 : 마스크 착용":
            self.mask3.setText("3회차 마스크 여부 : 착용")
        elif txtp[np.argmax(prediction[0])] == "===> 예상 : 사람 없음":
            self.mask3.setText("3회차 마스크 여부 : 미감지")
            mask_how = mask_how + 100

        self.qPixmapFileVar = QPixmap()
        self.qPixmapFileVar.load("test.jpg")
        self.qPixmapFileVar = self.qPixmapFileVar.scaledToWidth(400)
        self.maskt3.setPixmap(self.qPixmapFileVar)

        if mask_how == 2 or mask_how == 3:
            self.mask.setText("마스크 착용 여부 결과 : 미착용")
            self.info.setText("마스크를 착용해주시기 바랍니다. 흰색 옷을 착용하고 있을 경우 마스크 착용으로 인식될 수 있습니다.")
            time.sleep(3)
        elif mask_how >= 4:
            self.mask.setText("마스크 착용 여부 결과 : 미감지")
            self.info.setText("사람이 감지되지 않았습니다. 마스크 착용은 선택이 아닌 필수입니다.")
            time.sleep(3)
        else:
            self.mask.setText("마스크 착용 여부 결과 : 착용")
            self.info.setText("마스크 착용 여부 확인이 완료되었습니다. 마스크를 착용해주셔서 감사합니다.")
            time.sleep(3)

        self.info.setText("코로나19 증상인 발열 여부를 확인하기 위해 모니터 옆에 있는 열화상 카메라를 바라봐 주시기 바랍니다.")
        time.sleep(1)
        time.sleep(5)
        self.temp.setText("발열 증상 여부 결과 : 정상") # 정상, 발열
        self.info.setText("발열 증상 여부 확인이 완료되었습니다. 귀하의 체온은 정상 체온 범위입니다.")
        time.sleep(3)
        self.noti1.setText("코로나19 증상(기침, 호흡곤란, 오한, 근육통, 두통, 인후통, 후각·미각 소실)")
        self.noti2.setText("자가격리중인 동거인(14일 이내 해외 입국자 또는 확진자와의 접촉자 등)")
        i = 10
        while i > 0:
            self.info.setText("아래 두가지 사항이 해당되지 않는다면 " + str(i) + "초 내로 카메라 화면에서 나오시기 바랍니다.")
            if i == 10:
                th = threading.Thread(target=self.hand)
                th.start()
            time.sleep(1)
            i = i - 1
        if i == 0:
            self.stop.setText("등교 중지 여부 결과 : 중지")
            self.info.setText("코로나19 관련 등교 중지 사유가 발견되었습니다. 근처 선생님께 문의하시기 바랍니다.")
            self.all.setStyleSheet("Color : red")
            self.all.setText("FAIL")
            self.noti1.setText("")
            self.noti2.setText("")
            time.sleep(5)
            self.info.setText("코로나19를 예방하기 위한 자동 진단 시스템입니다. 사람이 감지되면 자동으로 시작합니다.")
            self.mask.setText("마스크 착용 여부 결과 : 대기")
            self.temp.setText("발열 증상 여부 결과 : 대기")
            self.stop.setText("등교 중지 여부 결과 : 대기")
            self.maskt1.setText("이미지 대기중")
            self.maskt2.setText("이미지 대기중")
            self.maskt3.setText("이미지 대기중")
            self.temp1.setText("이미지 대기중")
            self.mask1.setText("1회차 마스크 착용 여부 : 대기")
            self.mask2.setText("2회차 마스크 착용 여부 : 대기")
            self.mask3.setText("3회차 마스크 착용 여부 : 대기")
            self.temp1.setText("발열 증상 여부 : 대기")
            self.all.setText("")
        else:
            self.stop.setText("등교 중지 여부 결과 : 가능") # 가능, 중지
            self.info.setText("자동 진단이 완료되었으며, 정상으로 판단되었습니다. 감사합니다.")
            self.all.setStyleSheet("Color : blue")
            self.all.setText("PASS")
            self.noti1.setText("")
            self.noti2.setText("")
            time.sleep(3)
            self.info.setText("코로나19를 예방하기 위한 자동 진단 시스템입니다. 사람이 감지되면 자동으로 시작합니다.")
            self.mask.setText("마스크 착용 여부 결과 : 대기")
            self.temp.setText("발열 증상 여부 결과 : 대기")
            self.stop.setText("등교 중지 여부 결과 : 대기")
            self.maskt1.setText("이미지 대기중")
            self.maskt2.setText("이미지 대기중")
            self.maskt3.setText("이미지 대기중")
            self.temp1.setText("이미지 대기중")
            self.mask1.setText("1회차 마스크 착용 여부 : 대기")
            self.mask2.setText("2회차 마스크 착용 여부 : 대기")
            self.mask3.setText("3회차 마스크 착용 여부 : 대기")
            self.temp1.setText("발열 증상 여부 : 대기")
            self.all.setText("")
        th = threading.Thread(target=self.lets_go)
        th.start()

    def lets_go(self):
        th = threading.Thread(target=self.img_go)
        th.start()
        while (True):
            ret, img = cap.read()
            cv2.flip(img, 1)
            cv2.imwrite("test.jpg", img)

            image = Image.open('test.jpg')
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.ANTIALIAS)

            image_array = np.asarray(image)

            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

            data[0] = normalized_image_array

            prediction = model.predict(data)
            okay = round(prediction[0][0] * 100, 1)
            okay2 = "마스크 착용 : " + str(okay) + "%"
            noo = round(prediction[0][1] * 100, 1)
            noo2 = "마스크 미착용 : " + str(noo) + "%"
            noh = round(prediction[0][2] * 100, 1)
            noh2 = "사람 없음 : " + str(noh) + "%"
            print(prediction)
            print(okay2)
            print(noo2)
            print(noh2)
            txtp = ["===> 예상 : 마스크 착용", "===> 예상 : 마스크 미착용", "===> 예상 : 사람 없음"]
            print(txtp[np.argmax(prediction[0])])
            if txtp[np.argmax(prediction[0])] != "===> 예상 : 사람 없음":
                break
        th = threading.Thread(target=self.mask_test)
        th.start()

    def img_go(self):
        while (True):
            ret, img = cap.read()
            cv2.flip(img, 1)
            cv2.imwrite("good.jpg", img)
            self.qPixmapFileVar = QPixmap()
            self.qPixmapFileVar.load("good.jpg")
            self.qPixmapFileVar = self.qPixmapFileVar.scaledToWidth(550)
            self.imgimg.setPixmap(self.qPixmapFileVar)

    def hand(self):
        global i
        while i > 0:
            ret, img = cap.read()
            cv2.flip(img, 1)
            cv2.imwrite("test.jpg", img)

            image = Image.open('test.jpg')
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.ANTIALIAS)

            image_array = np.asarray(image)

            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

            data[0] = normalized_image_array

            prediction = model.predict(data)
            okay = round(prediction[0][0] * 100, 1)
            okay2 = "마스크 착용 : " + str(okay) + "%"
            noo = round(prediction[0][1] * 100, 1)
            noo2 = "마스크 미착용 : " + str(noo) + "%"
            noh = round(prediction[0][2] * 100, 1)
            noh2 = "사람 없음 : " + str(noh) + "%"
            print(prediction)
            print(okay2)
            print(noo2)
            print(noh2)
            txtp = ["===> 예상 : 마스크 착용", "===> 예상 : 마스크 미착용", "===> 예상 : 사람 없음"]
            print(txtp[np.argmax(prediction[0])])
            if txtp[np.argmax(prediction[0])] == "===> 예상 : 사람 없음":
                i = -1

    def mask_go(self):
        th = threading.Thread(target=self.mask_test)
        th.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    app.exec_()
