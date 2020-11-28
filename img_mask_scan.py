# keras_model.h5 파일 다운

import tensorflow.keras
from PIL import Image, ImageOps, ImageFont, ImageDraw
import numpy as np
import math
import time
import cv2

np.set_printoptions(suppress=True)

model = tensorflow.keras.models.load_model('keras_model.h5')

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
cap = cv2.VideoCapture(1)

how = 0

while (True):
    ret, img = cap.read()
    cv2.flip(img, 1)
    cv2.imwrite("test.jpg", img)
    # time.sleep(0.1)

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
        if how >= 2:
            print("마스크를 착용하지 않으신 것 같습니다. 마스크를 착용해주세요.")

            pill_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pill_image)
            x1, y1 = 40, 10
            text = '코로나19 확산 예방을 위한 마스크 착용 여부를 검사중입니다.'
            draw.text((x1, y1), text, font=ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf', 20), fill=(50, 50, 50))
            img = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)

            pill_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pill_image)
            x1, y1 = 20, 35
            text = '마스크 미착용이 확인되었습니다. 마스크를 착용해주시기 바랍니다.'
            draw.text((x1, y1), text, font=ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf', 20), fill=(255, 0, 0))
            img = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)

            now = time.strftime('%Y_%m_%d-%H_%M_%S')
            now = "C://covid_mask/" + now + ".jpg"
            cv2.imwrite(now, img)
        else:
            how = how + 1
            print("마스크 미착용 감지됨")

            pill_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pill_image)
            x1, y1 = 40, 10
            text = '코로나19 확산 예방을 위한 마스크 착용 여부를 검사중입니다.'
            draw.text((x1, y1), text, font=ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf', 20), fill=(50, 50, 50))
            img = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)

            pill_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pill_image)
            x1, y1 = 90, 35
            text = '마스크 착용 여부가 확인되었습니다. 감사합니다.'
            draw.text((x1, y1), text, font=ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf', 20), fill=(0, 0, 255))
            img = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)

    elif txtp[np.argmax(prediction[0])] == "===> 예상 : 마스크 착용":

        how = 0
        pill_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pill_image)
        x1, y1 = 40, 10
        text = '코로나19 확산 예방을 위한 마스크 착용 여부를 검사중입니다.'
        draw.text((x1, y1), text, font=ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf', 20), fill=(50, 50, 50))
        img = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)

        pill_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pill_image)
        x1, y1 = 90, 35
        text = '마스크 착용 여부가 확인되었습니다. 감사합니다.'
        draw.text((x1, y1), text, font=ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf', 20), fill=(0, 0, 255))
        img = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)

    elif txtp[np.argmax(prediction[0])] == "===> 예상 : 사람 없음":

        pill_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pill_image)
        x1, y1 = 40, 10
        text = '코로나19 확산 예방을 위한 마스크 착용 여부를 검사중입니다.'
        draw.text((x1, y1), text, font=ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf', 20), fill=(50, 50, 50))
        img = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)

        pill_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pill_image)
        x1, y1 = 190, 35
        text = '사람이 감지되지 않았습니다.'
        draw.text((x1, y1), text, font=ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf', 20), fill=(0, 255, 0))
        img = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)

    cv2.imshow('covid_mask', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        now = time.strftime('%Y_%m_%d-%H_%M_%S')
        now = "C://covid_mask/" + now + ".jpg"
        cv2.imwrite(now, img)
        break

cap.release()
cv2.destroyAllWindows()
