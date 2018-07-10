import cv2
import numpy as np 
import os

def check_path(path):
	dir = os.path.dirname(path)
	if not os.path.exists(dir):
		print("Please create trainning data first")
        
def check_id_name(path):
    if os.path.exists("dataset/.DS_Store"):
        os.remove("dataset/.DS_Store")
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    check_id = []
    check_name = []
    for imagePath in imagePaths:
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        name = str(os.path.split(imagePath)[-1].split(".")[2])
        check_id.append(id)
        check_name.append(name)
    check_id_name.id = list(set(check_id))
    check_id_name.name = list(set(check_name))

recognizer = cv2.face.LBPHFaceRecognizer_create()
f_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
check_path("trainer/")
check_id_name("dataset/")

recognizer.read('trainer/trainer.yml')

font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
cap.set(3, int(640))
cap.set(4, int(480))

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = f_detector.detectMultiScale(gray, 1.2,5)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x-20,y-20), (x+w+20,y+h+20), (255,0,0), 1)
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        name_id = " "
        #confidence = format(round(100 - confidence))
        if (confidence < 65):
            if any(Id == i for i in check_id_name.id):
                Id = check_id_name.id.index(Id)
                name_id = check_id_name.name[Id]
        else:
            name_id = name_id
        print(Id)
        print(confidence)
        cv2.rectangle(frame, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(frame, str(name_id), (x,y-40), font, 1, (255,255,255), 3)
        cv2.imshow("recognizer", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
