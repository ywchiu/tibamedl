from flask import Flask,jsonify,request
import keras
import numpy as np
import cv2 as cv
import cv2
import numpy
import matplotlib.pyplot as plt
from PIL import Image
import requests
import os
from keras.preprocessing import image

app=Flask(__name__)

import random

def getCaptcha(run):
    dest = 'digits2'
    res = requests.get('https://serv.gcis.nat.gov.tw/pub/kaptcha.jpg?code={}'.format(random.random()))
    with open('captcha.jpg', 'wb') as f:
        f.write(res.content)
    
    pil_image = Image.open('captcha.jpg').convert('L')
    open_cv_image = numpy.array(pil_image) 
    ret, thresh = cv2.threshold(open_cv_image, 127, 255, 0)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key=lambda x:x[1])
    ary = []
    for (c,_) in cnts:
        (x,y,w,h) = cv2.boundingRect(c)
        #print(x,y,w,h)
        if w >= 10 and w <= 25 and h >= 24:
            ary.append((x,y,w,h) )
    data = []
    for idx, (x,y,w,h) in enumerate(ary):
        fig = plt.figure()
        roi = open_cv_image[y:y+h, x:x+w]
        thresh = roi.copy()
        plt.imshow(thresh)
        plt.savefig(os.path.join(dest, '{}_{}.jpg'.format(run,idx)), dpi=100)

@app.route("/detect/", methods=['GET'])
def getKaptcha():
    res = []
    getCaptcha(1)
    clf = keras.models.load_model('captcha.h5')
    for f in os.listdir('digits2/'):
        fig = plt.figure()
        test_image = image.load_img('digits2/'+f, target_size= (60,40))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        predicted = clf.predict_classes(test_image)
        res.append(str(predicted[0]))
    a = ''.join(res)
    return jsonify({'data' : a})

if __name__=="__main__":
	app.run()