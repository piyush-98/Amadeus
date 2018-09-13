
# coding: utf-8

# In[5]:


import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json


# In[ ]:


import zipfile
zip_ref = zipfile.ZipFile('image.zip', 'r')
zip_ref.extractall()
zip_ref.close()


# In[8]:


from scipy.misc import imread
from scipy.misc import imresize
import os
ls=[]
folder='wwe'
files=os.listdir(folder)
files=list(map(lambda x: os.path.join(folder,x),files))
a=(len(files))
for i in range(a):
    im = imread(files[i])
    im=imresize(im,[32,32,3])
    im=im.flatten()
    ls.append(im)


# In[9]:


import numpy as np

raw_data=np.array(ls,dtype=object)
print(raw_data.shape)


# In[10]:


from sklearn.model_selection import train_test_split
train_data,test_data=train_test_split(raw_data,train_size=0.7,test_size=0.3,shuffle=True)
print(train_data.shape)


# In[97]:


x1=x_train[0:5000,:]
x2=x_train[5000:10000,:]
x1.shape,x2.shape


# In[98]:


x1=x1.reshape(5000,3072)
x2=x2.reshape(5000,3072)
x1.shape,x2.shape


# In[99]:


y1=y_train[0:5000]
y2=y_train[5000:10000]
y1.shape,y2.shape


# In[102]:


beg=5000
end=len(x1) + len(train_data)
#print(end)
images=np.zeros(shape=[end+1,3072])
images[0:beg, :]=x1
images[beg:end,:]=train_data
#images[end]=images[11330]
print(images.shape)


# In[ ]:


images=images.reshape(-1,32,32,3)
images=images/255.0
#print(images[2])


# In[104]:


images.shape


# In[105]:


tes=x2[0:5000,:]
print(tes.shape)
end=len(tes) + len(test_data)
test_images=np.zeros(shape=[end,3072])
beg=5000
test_images[0:beg,:]=tes
test_images[beg:end,:]=test_data
print(test_images[5201])
print(test_images.shape)


# In[106]:


test_images=test_images.reshape(-1,32,32,3)
print(test_images.shape)
test_images=test_images/255.0


# In[107]:


test_images.shape


# In[109]:


label_train=np.zeros(shape=[6660,1])
label_train[5000:6660,:]=1
print(label_train.shape)


# In[110]:


label_test=np.zeros(shape=[5711,1])
label_test[5000:5711,:]=1
print(label_test[5001])


# In[111]:


a=(images.shape[0])
i=np.random.permutation(a)
x_train1,y_train1=images[i],label_train[i]
print(y_train1.shape,x_train1.shape)


# In[112]:


j=np.random.permutation(test_images.shape[0])
data_test,label_test=test_images[j],label_test[j]
print(data_test.shape,label_test.shape)


# In[ ]:


num_classes=2
y_train_main = keras.utils.to_categorical(y_train1, num_classes)
y_test_main = keras.utils.to_categorical(label_test, num_classes)


# In[114]:


y_train_main.shape,y_test_main.shape


# In[ ]:


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train1.shape[1:]))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


# In[ ]:


opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# In[117]:


batch_size=32
epochs=12
model.fit(x_train1, y_train_main,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(data_test, y_test_main),
              shuffle=True)


# In[118]:


scores = model.evaluate(data_test, y_test_main, verbose=1)
scores


# In[ ]:


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# In[120]:


model.save_weights("model.h5")
print("Saved model to disk")


# In[ ]:


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


# In[122]:


loaded_model.load_weights("model.h5")
print("Loaded model from disk")


# In[145]:


loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.predict_classes(im, verbose=1)

