import pickle
import numpy as np
import cv2
import mediapipe as mp
import pymysql
from sklearn.ensemble import RandomForestClassifier
drawingutils=mp.solutions.drawing_utils
drawingstyle=mp.solutions.drawing_styles
video=cv2.VideoCapture(0)
handrecognition=mp.solutions.hands
connection=pymysql.connect(
    host="localhost",
    user="root",
    password="password",
    database="signlanguage",
)
def landmarks_to_array(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
cursor=connection.cursor()
model=RandomForestClassifier()
cursor.execute("SELECT sign_encoding FROM signs")
features_train=cursor.fetchall()
cursor.execute("SELECT sign_name FROM signs")
variable_train=cursor.fetchall()
print(type(features_train), type(variable_train))
features_train = [landmarks_to_array(pickle.loads(item[0])) for item in features_train]
variable_train = [pickle.loads(item[0]) for item in variable_train]
model.fit(features_train,variable_train)
with handrecognition.Hands(max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5,model_complexity=1) as hands:
    while True:
        success,img=video.read()
        if not success:
            break
        results=hands.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                drawingutils.draw_landmarks(img,landmark_list=landmarks,connections=handrecognition.HAND_CONNECTIONS)
                h, w, _ = img.shape
                xcords=[int(lm.x*w) for lm in landmarks.landmark]
                ycords=[int(lm.y*h) for lm in landmarks.landmark]
                xmax,ymax=max(xcords),max(ycords)
                xmin,ymix=min(xcords),min(ycords)
                cv2.rectangle(img, (xmin, ymix), (xmax, ymax), (0, 255, 0), 2)
                prediction_input = landmarks_to_array(landmarks).reshape(1, -1)
                prediction = model.predict(prediction_input)
                cv2.putText(img,f"{prediction}", (xmax, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if cv2.waitKey(10)==ord("q"):
            break
        cv2.imshow("VIDEO OUTPUT",cv2.flip(img,1))
video.release()
cv2.destroyAllWindows()