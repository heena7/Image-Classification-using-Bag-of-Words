# In[1]:

import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime


# In[99]:Listing all the directories

train = os.listdir("Data/train")
train_names = ([x for x in train
           if not (x.startswith('.'))])
print(train_names)


train_path = "Data/train"
i_imagePath = []

i_classes = np.array(['labels'],dtype = 'object')
surf = cv2.xfeatures2d.SURF_create(400)
class_id = 0
des_list = []


from matplotlib import pyplot as plt
print(datetime.now())
for train_name in train_names:
    dir = os.path.join(train_path,train_name)
    im = os.listdir(dir)

    for p in im:
        class_path = train_path +"/"+train_name+"/"+p
        image = cv2.imread(class_path)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        kp = surf.detect(gray,None)
        kp,des = surf.compute(gray,kp)
        #print(len(kp))
        img = cv2.drawKeypoints(gray,kp,image)
        des_list.append((class_path,des))
        i_imagePath.append(class_path)
        i_classes = np.vstack([i_classes,train_name])
    class_id += 1
        #plt.imshow(img)
print(class_id)
print(len(im))


# In[100]:

descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor)) 
print (descriptors.shape)    


# In[101]:

k = 100
from scipy.cluster.vq import kmeans
voc, variance = kmeans(descriptors, k, 1) 
print(voc.size)
print(variance)


# In[110]:

im_features = np.zeros((len(i_imagePath), k), "float32")


# In[111]:

from scipy.cluster.vq import vq

for i in range(len(i_imagePath)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1
print(im_features)


# In[112]:

for i in range(len(i_imagePath)):
    im_features[i] = (im_features[i]/sum(im_features[i]))
    print(im_features[i])



nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(i_imagePath)+1) / (1.0*nbr_occurences + 1)), 'float32')


# In[106]:

stdSlr = StandardScaler().fit(im_features)


# In[107]:

im_features = stdSlr.transform(im_features)


# In[113]:

clf = LinearSVC()
print(np.array(i_classes[1:]))


# In[114]:

clf.fit(im_features, np.ravel(np.array(i_classes[1:])))


# In[115]:

test = os.listdir("Data/val")
test_names = ([x for x in train
           if not (x.startswith('.'))])


# In[116]:

test_path = "Data/val"


# In[126]:

test_classes = np.array(['labels'],dtype = 'object')
surf = cv2.xfeatures2d.SURF_create(400)
class_id = 0
dest_list = []
ii_imagePath = []

for test_name in test_names:
    dir = os.path.join(test_path,test_name)
    im = os.listdir(dir)

    for p in im:
        class_path = test_path +"/"+test_name+"/"+p
        image = cv2.imread(class_path)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        kp = surf.detect(gray,None)
        kp,des = surf.compute(gray,kp)
        #print(len(kp))
        img = cv2.drawKeypoints(gray,kp,image)
        dest_list.append((class_path,des))
        ii_imagePath.append(class_path)
        #i_classes = np.vstack([i_classes,train_name])
        test_classes = np.vstack([test_classes,test_name])


descriptors = dest_list[0][1]
for image_path, descriptor in dest_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))
k = 100
from scipy.cluster.vq import kmeans
voc, variance = kmeans(descriptors, k, 1) 
imt_features = np.zeros((len(ii_imagePath), k), "float32")
from scipy.cluster.vq import vq

for i in range(len(ii_imagePath)):
    words, distance = vq(dest_list[i][1],voc)
    for w in words:
        imt_features[i][w] += 1

imt_features[0] = (imt_features[0]/sum(imt_features[0]))
print(imt_features[0])
        
nbr_occurences = np.sum( (imt_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(ii_imagePath)+1) / (1.0*nbr_occurences + 1)), 'float32')


# In[127]:
#import pandas as pd
#from pandas import DataFrame
#df=pd.DataFrame(data=0, index = np.arange(3823), columns = np.arange(65))
clf.predict(imt_features[:100])


# In[128]:

print(im_features)


# In[130]:

from sklearn.metrics import accuracy_score
print(accuracy_score(clf.predict(imt_features),test_classes[1:]))


# In[ ]: