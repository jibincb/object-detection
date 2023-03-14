import numpy as np
import cv2

image_path = 'IMG_20201126_181055.jpg'
prototxt_path = 'models/MobileNetSSD_deploy.prototxt.txt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.2

classes = ["background","aeroplane",'bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

np.random.seed(543210)
colors = np.random.uniform(0,255,size=(len(classes),3))
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
image = cv2.imread(image_path)
height,weight = image.shape[0], image.shape[1]
blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),0.007,(300,300),130)

net.setInput(blob)
detect_objects = net.forward()
print(detect_objects[0][0][1])