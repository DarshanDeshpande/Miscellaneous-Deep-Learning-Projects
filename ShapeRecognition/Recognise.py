import cv2
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("ModelBest.h5")

def preprocess(img_path):
    x = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x,(150,150),interpolation = cv2.INTER_AREA)
    x=x/257
    return x


testlist,test = [],[]
for i in glob.glob('/content/*.png'):
  testlist.append(os.path.basename(i).split('.')[0])
  x = preprocess(i)
  plt.imshow(x)
  plt.show()
  test.append(x)

test = np.array(test)
test = test.reshape((-1,150,150,1))

classes = ['circle','rectangle','square','triangle','trapezium']

answers=[]
preds = model.predict_classes(test,verbose=1)
for i in preds:
  answers.append(classes[i])

print(answers)
