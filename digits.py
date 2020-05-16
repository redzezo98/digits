import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import pickle





#===============================================================
path = 'myData'
test_rate = 0.2
val_rate = 0.2
imageDimensions = (32,32,3)

batchSizeVal = 50

epochsVal = 10

stepsPerEpochval = 2000

#===============================================================


images = []
classNum = []
myList = os.listdir(path)
print("Total Number of classes detected:",len(myList))
numberOfClasses = len(myList)
print("Importing Classes .....")

for x in range (0,numberOfClasses):
    myPiclist = os.listdir(path + "/" + str(x))
    for y in myPiclist:
        curImg = cv2.imread(path + "/" + str(x) + "/" + y)
        curImg = cv2.resize(curImg,(32,32))
        images.append(curImg)
        classNum.append(x)
    print(x,end=" ")
print(" ")

images = np.array(images)
classNum = np.array(classNum)

print(images.shape)
#print(classNum.shape)

# =========== Spliting the data ============

X_train,X_test,y_train,y_test = train_test_split(images,classNum,test_size=test_rate)
X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train,test_size=val_rate)

print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

numOfSamples = []
for x in range(0,numberOfClasses):
    #print(len(np.where(y_train==x)[0]))
    numOfSamples.append(len(np.where(y_train==x)[0]))
print(numOfSamples)

plt.figure(figsize=(10,5))
plt.bar(range(0,numberOfClasses),numOfSamples)
plt.title("Number of Images for each Class")
plt.xlabel("Class ID")
plt.ylabel("Number of images")
plt.show()

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img
#img = preProcessing(X_train[30])
#img = cv2.resize(img,(300,300))

#cv2.imshow("preProcessing",img)
#cv2.waitKey(0)


X_train= np.array(list(map(preProcessing,X_train)))
X_test= np.array(list(map(preProcessing,X_test)))
X_validation= np.array(list(map(preProcessing,X_validation)))


#### RESHAPE IMAGES
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)

#### IMAGE AUGMENTATION
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)

dataGen.fit(X_train)


y_train = to_categorical(y_train,numberOfClasses)
y_test = to_categorical(y_test,numberOfClasses)
y_validation = to_categorical(y_validation,numberOfClasses)


def myModel():
    numOfFilters = 60
    sizeOfFilters1 = (5,5)
    sizeOfFilters2= (3,3)
    sizeOfPool=(2,2)
    numofNode = 500

    model = Sequential()
    model.add((Conv2D(numOfFilters,sizeOfFilters1,input_shape=(imageDimensions[0],
                                                               imageDimensions[1],
                                                               1),activation='relu')))
    model.add((Conv2D(numOfFilters, sizeOfFilters1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(numOfFilters//2, sizeOfFilters2, activation='relu')))
    model.add((Conv2D(numOfFilters//2, sizeOfFilters2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(numofNode,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numberOfClasses,activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = myModel()

print(model.summary())

batchSizeVal = 50

epochsVal = 10

stepsPerEpochval = 2000

history = model.fit_generator(dataGen.flow(X_train,y_train,
                                 batch_size=batchSizeVal),
                                 steps_per_epoch=stepsPerEpochVal,
                                 epochs=epochsVal,
                                 validation_data=(X_validation,y_validation),
                                 shuffle=1)


plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')

#==================================

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy '])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

score = model.evaluate(X_test,y_test,verbose=0)
print('Test Score = ',score[0])
print('Test Accuracy = ',score[1])

pickle_out = open("model_trained.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()





