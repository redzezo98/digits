import numpy as np
import cv2
import pickle

###################################################
width = 640
height = 480
threshold = 0.8

###################################################




pickle_in = open("model_trained_10.p","rb")
model = pickle.load(pickle_in)


def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img


imgOriginal = cv2.imread('test_5.jpg')
img = np.asarray(imgOriginal)
img = cv2.resize(img,(32,32))
img = preProcessing(img)
cv2.imshow("Processed img",img)
img = img.reshape(1,32,32,1)

classIndex = int(model.predict_classes(img))
print(classIndex)

predictions = model.predict(img)
probVal = np.amax(predictions)

if probVal >threshold:
    cv2.putText(imgOriginal,str(classIndex) + "  " +str(probVal),(25,25),cv2.FONT_HERSHEY_COMPLEX,
                1,(0,0,255),1)

cv2.imshow("Original Image",imgOriginal)
cv2.waitKey(0)

