import cv2
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import time
import os


mixer.init()
#sound = mixer.Sound('DrowsinessDetection\\alarm.wav')
sound = mixer.Sound(
    'C:\\Users\\Saurav\\OneDrive\\Desktop\\minor_project\\DrowsinessDetection\\alarm.wav')

path = os.getcwd()
face_detector = cv2.CascadeClassifier(
    'C:\\Users\\Saurav\\OneDrive\\Desktop\\minor_project\\DrowsinessDetection\\haar cascade files\\haarcascade_frontalface_alt.xml')
leye_detector = cv2.CascadeClassifier(
    'C:\\Users\\Saurav\\OneDrive\\Desktop\\minor_project\\DrowsinessDetection\\haar cascade files\\haarcascade_lefteye_2splits.xml')
reye_detector = cv2.CascadeClassifier(
    'C:\\Users\\Saurav\\OneDrive\\Desktop\\minor_project\\DrowsinessDetection\\haar cascade files\\haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

eye_model = load_model(
    'C:\\Users\\Saurav\\OneDrive\\Desktop\\minor_project\\DrowsinessDetection\\models\\eyemodel.h5')
yawn_model = load_model(
    'C:\\Users\\Saurav\\OneDrive\\Desktop\\minor_project\\DrowsinessDetection\\models\\yawnmodel.h5')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 500)
cap.set(4, 500)
cap.set(5, 10)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2
rpred = [99]
lpred = [99]
yawn_pred = 0
ptime = 0
yawn_score = 0


def predictor_function(img, roi, i, j, predictor):
    value = 0
    for (x, y, w, h) in roi:
        pred_img = img[y:y+h, x:x+w]
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)
        pred_img = cv2.resize(pred_img, (i, j))
        pred_img = pred_img/255
        pred_img = pred_img.reshape(i, j, 1)
        pred_img = np.expand_dims(pred_img, axis=0)
        pred_value = predictor.predict(pred_img)
        value = pred_value[0][0]
    return value


while(True):
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))

    left_eye = leye_detector.detectMultiScale(gray)
    right_eye = reye_detector.detectMultiScale(gray)

    reye_pred = predictor_function(frame, right_eye, 24, 24, eye_model)
    leye_pred = predictor_function(frame, left_eye, 24, 24, eye_model)
    yawn_pred = predictor_function(frame, faces, 64, 64, yawn_model)

    cv2.putText(frame, "yawning per:"+str(yawn_pred),
                (10, height-40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "right eye per:"+str(reye_pred),
                (10, height-60), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "left eye per:"+str(leye_pred),
                (10, height-80), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if(reye_pred > 0.5 and leye_pred >= 0.5):
        score = score-1
        cv2.putText(frame, "Open ", (10, height-20), font,
                    1, (255, 255, 255), 1, cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score = score+1
        cv2.putText(frame, "Closed", (10, height-20), font,
                    1, (255, 255, 255), 1, cv2.LINE_AA)
    if(yawn_pred >= 0.5):
        cv2.putText(frame, "Yawning", (100, height-20),
                    font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Not Yawning", (100, height-20),
                    font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if(score < 0):
        score = 0
    cv2.putText(frame, 'Score:'+str(score), (300, height-20),
                font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    cv2.putText(frame, f'fps: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    if(score > 15):
        # person is feeling sleepy so we beep the alarm
        # cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
        except:  # isplaying = False
            pass
    # if yawn_pred >= 0.8:
    #     cv2.putText(frame, 'yawning' , (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    # else:
    #     cv2.putText(frame, 'not yawning' , (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
