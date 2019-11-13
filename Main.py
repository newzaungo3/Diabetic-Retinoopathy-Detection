import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import glob
import pywt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.utils import to_categorical
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


import warnings
warnings.filterwarnings("ignore")
train_image = []
test_image = []
df_train = pd.read_csv('trainlabel/trainLabels.csv')
y = df_train['level']
score_svm = []
score_rf = []
test = "15_right.jpeg"

def getscore(model,x_train, x_test, y_train, y_test):
    model.fit(x_train,y_train)
    return model.score(x_test,y_test)
#tqdm(range(df_train.shape[0]))

fig = plt.figure(figsize=(25, 16))
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
for i in tqdm(range(50)) :
    img = image.load_img('train/'+'trainimage/'+'train/'+df_train['image'][i]+'.jpeg',target_size=(224,224,3))
    img = image.img_to_array(img)
    img = img / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = exposure.equalize_hist(img)
    img_adpeq = exposure.equalize_adapthist(img,clip_limit=0.03)
    train_image.append(img)

image_test = cv2.imread(test)
image_test = cv2.resize(image_test,(224,224))
l_channel, a_channel, b_channel = cv2.split(image_test)
image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)
cl1 = clahe.apply(a_channel)
test_image.append(cl1)

X = np.asarray(train_image)
plt.imshow(X[0])
plt.show()
print(X.shape)
Xtest = np.asarray(test_image)
nsamples, nx, ny = Xtest.shape
X2 =Xtest.reshape((nsamples,nx*ny))


kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index],X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    trainSize= len(X_train)
    testSize = len(X_test)
    X_train =X_train.reshape((trainSize,-1))
    X_test = X_test.reshape((testSize,-1))
    #SVM
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm = SVC()
    svm.fit(X_train,y_train)

    clf_svm_scaled = SVC()
    clf_svm_scaled.fit(X_train_scaled, y_train)

    y_pred = svm.predict(X_test)
    print(y_pred)
    y_pred_scaled = clf_svm_scaled.predict(X_test_scaled)
    print('Non Standardize')
    print('Accuracy of SVM classifier on training set: {:.4f}'
          .format(svm.score(X_train, y_train) * 100))
    print('Accuracy of SVM classifier on test set: {:.4f}'
          .format(svm.score(X_test, y_test) * 100))
    print('\nStandardize')
    print('Accuracy of SVM classifier on training set: {:.4f}'
          .format(clf_svm_scaled.score(X_train_scaled, y_train) * 100))
    print('Accuracy of SVM classifier on test set: {:.4f}'
          .format(clf_svm_scaled.score(X_test_scaled, y_test) * 100))

    #print(classification_report(y_test, y_pred))
    #print(confusion_matrix(y_test, y_pred))


    #score_svm.append(getscore(SVC(),X_train,X_test,y_train,y_test))
    #score_rf.append(getscore(RandomForestClassifier(),X_train,X_test,y_train,y_test))

#print("SVM")
#print(score_svm)
#print("Random Forest")
#print(score_rf)
