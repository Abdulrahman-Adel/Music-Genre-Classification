# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 20:24:00 2020

@author: Abdelrahman
"""

import pandas as pd
import numpy as np
import cv2
import os
import random

main_dir = "Data\images_original"

df = []

for filename in os.listdir(main_dir):
    genre = os.path.join(main_dir,filename)
    for img_name in os.listdir(genre):
        img_arr = cv2.imread(os.path.join(genre,img_name),cv2.IMREAD_GRAYSCALE)
        df.append([img_arr,filename])
        
random.shuffle(df) 

"""cv2.imshow("radnom_pic",random.choice(df)[0])

cv2.waitKey(0)
cv2.destroyAllWindows()"""       
        
X = []
y = []

for img, label in df:
    X.append(img)
    y.append(label)
    
X = np.array(X).reshape(-1,288,432,1)
y = np.array(y) 

y = pd.get_dummies(y)    

X = X/255 
    
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model = Sequential()

model.add(Conv2D(32,(3,3),activation="relu",input_shape=X.shape[1:]))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(512,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10,activation="softmax"))

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

history = model.fit(X_train,y_train,
                    batch_size=16,
                    epochs=10,
                    validation_data=(X_test,y_test),
                    callbacks=[es])

        
                