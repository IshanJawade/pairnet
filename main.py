
from flask import Flask, render_template, Response,  request, session, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

import os
from os.path import join, dirname, realpath
import sys
import cv2
# import the necessary packages
import numpy as np
import argparse
import time
from io import BytesIO
import io
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


# construct the argument parse and parse the arguments

confthres = 0.1
nmsthres = 0.1
yolo_path = './'

def get_labels(labels_path):
    # load the COCO class labels our YOLO model was trained on
    #labelsPath = os.path.sep.join([yolo_path, "yolo_v3/coco.names"])
    lpath=os.path.sep.join([yolo_path, labels_path])
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS

def get_colors(LABELS):
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    return COLORS

def get_weights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath

def load_model(configpath,weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net


def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format='PNG')
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr

def crop_it_bruh(centerX, centerY, width, height, dw, dh, filename, img):
  x = centerX
  y = centerY
  w = width
  h = height
  # converting YOLO Coordinates into normal from 
  l = int((x - w / 2) * dw)
  r = int((x + w / 2) * dw)
  t = int((y - h / 2) * dh)
  b = int((y + h / 2) * dh)
  if l < 0:
    l = 0
  if r > dw - 1:
    r = dw - 1
  if t < 0:
    t = 0
  if b > dh - 1:
    b = dh - 1
  #to crop images
  #cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 3)
  #crop_img = img[l+130:r+139, t+103:b+137] OPENCV cropping 
  cropped = img.crop((l,t,r,b))
  cropped.save( cropped_path +filename+'.jpg')



def get_predection(image,net,LABELS,COLORS,filename, img):
    (H, W) = image.shape[:2]
    print("111111111111111111111111111111111111111111111")
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    print("22222222222222222222222222222222222222222222222")
                             
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            print(boxes)
            print(classIDs)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
            dw, dh, _ = image.shape
            # crop_it_bruh(centerX, centerY, width, height, dw, dh, filename,img)
            crop_img = image[y:y+h, x:x+w]
            return image, crop_img
    
    return image, []


labelsPath="yolo_v3/coco.names"
cfgpath="yolo_v3/yolov3_retina.cfg"
wpath="yolo_v3/yolov3_retina.weights"
Lables=get_labels(labelsPath)
CFG=get_config(cfgpath)
Weights=get_weights(wpath)
nets=load_model(CFG,Weights)
Colors=get_colors(Lables)
output_path = './static/detections/'
cropped_path = './static/cropped/'
gray_path = './static/gray/'

app = Flask(__name__, static_url_path='/static')
UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'static/uploads/..')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
GRAY_FOLDER = join(dirname(realpath(__file__)), 'static/gray/..')
app.config['GRAY_FOLDER'] = GRAY_FOLDER 


# Initialize the Flask application

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/about")
def about():
  return render_template("about.html")

@app.route("/team")
def team():
  return render_template("team.html")

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
       # create a secure filename
      filename = secure_filename(f.filename)
      print(filename)
      # save file to /static/uploads
      filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      print(filepath)
      f.save(filepath)
      img = Image.open(filepath)
      npimg=np.array(img)
      image=npimg.copy()
      #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
      image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
      print(image.shape)
      #######
      # cv2.imwrite(gray_path + '{}' .format(filename), gray_image)
      # filepath = os.path.join(app.config['GRAY_FOLDER'], filename)
      # print(filepath+ "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      # gray_img = Image.open(filepath)
      # npimg=np.array(gray_img)
      # gray_img=npimg.copy()
      #######
      res, crop_img = get_predection(image,nets,Lables,Colors,filename,img)
      cv2.imwrite(output_path + '{}' .format(filename), res)
      if(crop_img!=[]):
        cv2.imwrite(cropped_path + '{}' .format(filename), crop_img)
      print('output saved to: {}'.format(output_path +  '{}'.format(filename)))
      print(filename)
      return render_template("uploaded.html", display_detection = filename, fname = filename)    
   
   
    # start flask app
if __name__ == '__main__':
   app.run(port=4200, debug=True)
