import cv2
import mediapipe as mp
import pymysql
import pickle
import time
connection=pymysql.connect(
    host="localhost",
    user="root",
    password="password",
    database="signlanguage",
)
cursor=connection.cursor()
drawingutils=mp.solutions.drawing_utils
drawingstyle=mp.solutions.drawing_styles
video=cv2.VideoCapture(0)
alphabet=input("Enter alphabet/phrase for the model to be trained on:")
handrecognition=mp.solutions.hands
with handrecognition.Hands(max_num_hands=1,model_complexity=1,min_tracking_confidence=0.5,min_detection_confidence=0.8) as hands:
    while True:
        success,image=video.read()
        if not success:
            break
        results=hands.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                drawingutils.draw_landmarks(image,landmark_list=landmarks,connections=handrecognition.HAND_CONNECTIONS)
                signature=time.time()
                cursor.execute("INSERT INTO signs(sign_name,sign_encoding,signature) VALUES(%s,%s,%s)",(pickle.dumps(alphabet),pickle.dumps(landmarks),str(signature)))
        cv2.imshow("VIDEO",cv2.flip(image,1))

        if cv2.waitKey(10)==ord("q"):
            connection.commit()
            break
video.release()
cv2.destroyAllWindows()

