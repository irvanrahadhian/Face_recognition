import cv2
import os
import numpy as np
from PIL import Image

def check_path(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

recognizer = cv2.face.LBPHFaceRecognizer_create()
f_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def ImagesAndLabels(path):
	if os.path.exists("dataset/.DS_Store"):
		os.remove("dataset/.DS_Store")

	imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
	f_samples=[]
	f_ids=[]
	for imagePath in imagePaths: #get every image
		fetch_img = Image.open(imagePath).convert('L')
		image_np = np.array(fetch_img,'uint8')
		id = int(os.path.split(imagePath)[-1].split(".")[1])
		faces = f_detector.detectMultiScale(image_np)

		for (x,y,w,h) in faces:
			#add image and id
			f_samples.append(image_np[y:y+h,x:x+w])
			f_ids.append(id)
	return f_samples, f_ids #pass face array and id array
	

faces,f_ids = ImagesAndLabels('dataset')
recognizer.train(faces, np.array(f_ids)) #train the model using faces and ids
check_path('trainer/')
recognizer.save('trainer/trainer.yml')