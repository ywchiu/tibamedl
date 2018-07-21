from flask import Flask,jsonify,request
import keras
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import requests
from keras.preprocessing import image

face_cascade = cv.CascadeClassifier('C:\\ProgramData\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')



app=Flask(__name__)

@app.route("/detect/", methods=['POST'])
def getSalary():
	if request.method=='POST':
		url =request.form['url']
		res = requests.get(url)
		
		with open('test.jpg', 'wb') as f:
			f.write(res.content)
		img = cv.imread('test.jpg')
		faces = face_cascade.detectMultiScale(img, 1.3, 5)
		im = Image.open('test.jpg')
		x,y,w,h = faces[0]
		box = (x, y, x+w, y+h)
		crpim = im.crop(box).resize((64,64))
		crpim.save('test2.jpg')
		test_image = image.load_img('test2.jpg', target_size= (64,64))
		test_image = image.img_to_array(test_image)
		test_image = np.expand_dims(test_image, axis = 0)
		clf = keras.models.load_model('idol.pkl')
		predicted = clf.predict_classes(test_image)
		return jsonify({'data' : str(predicted[0]) })

if __name__=="__main__":
	app.run()