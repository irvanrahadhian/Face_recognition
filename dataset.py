import cv2
import os

def check_path(path): #check path for directory and check id for registered person
    if os.path.exists("dataset/.DS_Store"):
        os.remove("dataset/.DS_Store")
    check_path.f_ids = []
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        check_path.f_ids.append(id)

    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def check_id(): #check id for new person
    while True:
        check_id.f_id = input('enter your id :')
        if any(int(check_id.f_id) == i for i in check_path.f_ids):
            print("Id already used, please enter another one")
        else:
            return False        

cap = cv2.VideoCapture(0)
f_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
check_path("dataset/")
check_id()
f_name = input('enter your name :')
count = 0 #init sample face image

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = f_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces: #loop each faces
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2 )
        count += 1
        cv2.imwrite("dataset/User." + str(check_id.f_id) +'.'+ str(f_name) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('feed', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    elif count > 20:
        break

cap.release()
cv2.destroyAllWindows()