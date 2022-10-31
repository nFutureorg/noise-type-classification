from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import random
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import config

# make a fix file size
fixed_size  = tuple((1024, 600))

#train path 
train_path = "dataset/train"

# no of trees for Random Forests
num_tree = 100

# bins for histograms 
bins = 8

# train_test_split size
test_size = 0.10

# seed for reproducing same result 
seed = 9 

# features description -1:  Hu Moments

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor -2 Haralick Texture 

def fd_haralick(image):
    # conver the image to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Ccompute the haralick texture fetature ve tor 
    haralic = mahotas.features.haralick(gray).mean(axis=0)
    return haralic

# feature-description -3 Color Histogram

def fd_histogram(image, mask=None):
    # conver the image to HSV colors-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #COPUTE THE COLOR HISTPGRAM
    hist  = cv2.calcHist([image],[0,1,2],None,[bins,bins,bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist,hist)
    # return the histog....
    return hist.flatten()

# get the training data labels 
train_labels = os.listdir(train_path)

# sort the training labesl 
train_labels.sort()
print(train_labels)

# empty list to hold feature vectors and labels 
global_features = []
labels = []

i, j = 0, 0 
k = 0

# num of images per class 
images_per_class = 278


# ittirate the folder to get the image label name

#%time
# lop over the training data sub folder 

for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

    k = 1
    # loop over the images in each sub-folder
        
    for file in os.listdir(dir):

        file = dir + "/" + os.fsdecode(file)
       
        # read the image and resize it to a fixed-size
        image = cv2.imread(file) 
        
        if image is not None:
            image = cv2.resize(image,fixed_size)
            fv_hu_moments = fd_hu_moments(image)
            fv_haralick   = fd_haralick(image)
            fv_histogram  = fd_histogram(image)
        #else:
            #print("image not loaded")
                
        #image = cv2.imread(file)        
        #image = cv2.resize(image,fixed_size)

        # Concatenate global features
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

        i += 1
        k += 1
    print("[STATUS] processed folder: {}".format(current_label))
    j += 1

print("[STATUS] completed Global Feature Extraction...")

#%time
# get the overall feature vector size
print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# get the overall training label size
print("[STATUS] training Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print("[STATUS] training labels encoded...{}")
# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("[STATUS] feature vector normalized...")

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))

# save the feature vector using HDF5
h5f_data = h5py.File('output/data.h5', 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File('output/labels.h5', 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] end of training..")


# import the feature vector and trained labels

h5f_data = h5py.File('output/data.h5', 'r')
h5f_label = h5py.File('output/labels.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)


trainDataGlobal = np.array(global_features)

trainLabelsGlobal = np.array(global_labels)





# get the training data labels 
train_labels = os.listdir('dataset/validation')

# sort the training labesl 
train_labels.sort()
print(train_labels)

# empty list to hold feature vectors and labels 
global_features = []
labels = []

i, j = 0, 0 
k = 0

# num of images per class 
images_per_class = 278


# ittirate the folder to get the image label name

#%time
# lop over the training data sub folder 

for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

    k = 1
    # loop over the images in each sub-folder
        
    for file in os.listdir(dir):

        file = dir + "/" + os.fsdecode(file)
       
        # read the image and resize it to a fixed-size
        image = cv2.imread(file) 
        
        if image is not None:
            image = cv2.resize(image,fixed_size)
            fv_hu_moments = fd_hu_moments(image)
            fv_haralick   = fd_haralick(image)
            fv_histogram  = fd_histogram(image)
        #else:
            #print("image not loaded")
                
        #image = cv2.imread(file)        
        #image = cv2.resize(image,fixed_size)

        # Concatenate global features
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

        i += 1
        k += 1
    print("[STATUS] processed folder: {}".format(current_label))
    j += 1

print("[STATUS] completed Global Feature Extraction...")

#%time
# get the overall feature vector size
print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# get the overall training label size
print("[STATUS] training Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print("[STATUS] training labels encoded...{}")
# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("[STATUS] feature vector normalized...")

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))

# save the feature vector using HDF5
h5f_data = h5py.File('output/data_test.h5', 'w')
h5f_data.create_dataset('dataset_1_test', data=np.array(rescaled_features))

h5f_label = h5py.File('output/labels_test.h5', 'w')
h5f_label.create_dataset('dataset_1_test', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] end of training..")


# import the feature vector and trained labels

h5f_data = h5py.File('output/data_test.h5', 'r')
h5f_label = h5py.File('output/labels_test.h5', 'r')

global_features_string = h5f_data['dataset_1_test']
global_labels_string = h5f_label['dataset_1_test']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)


testDataGlobal = np.array(global_features)

testLabelsGlobal = np.array(global_labels)




# create all the machine learning models
models = []
models.append(('LR', LogisticRegression(random_state=seed)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=seed)))
models.append(('RF', RandomForestClassifier(n_estimators=num_tree, random_state=seed)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=seed)))

# variables to hold the results and names
results = []
names   = []


# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=10,shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
#pyplot.show()
plt.savefig('com_ml.pdf', format='pdf', dpi=300)
plt.savefig('com_ml.png', format='png', dpi=300)
plt.close()


# create the model - Random Forests
clf  = RandomForestClassifier(n_estimators=100)

# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)

#print(clf.fit(trainDataGlobal, trainLabelsGlobal))

#clf_pred = clf.predict(testDataGlobal)
#clf_pred = clf.predict(global_feature.reshape(1,-1))[0]
#print(classification_report(testLabelsGlobal,clf_pred))
#print(confusion_matrix(trainLabelsGlobal,clf_pred))

#print(clf.predict(trainDataGlobal))



# path to test data
test_path = "dataset/test"

# get the training data labels 
test_labels = os.listdir(test_path)

# sort the training labesl 
test_labels.sort()
print(test_labels)



test_image = glob.glob(test_path + "/gaussian/*.jpg")
#test_image = random.choices(test_image)

test_image_w = glob.glob(test_path + "/poisson/*.jpg")
#test_image_w = random.choices(test_image_w)

test_image_m = glob.glob(test_path + "/mixed/*.jpg")
#test_image_m = random.choices(test_image_m)


for i in test_image_w:
  test_image.append(i)

for i in test_image_m:
  test_image.append(i)

# loop through the test images
#for file in glob.glob(test_path + "/*.jpg"):
#for f in 
count = 0
truelabels = []
predicted_labels = []
for file in test_image:    

    #file = test_path+'/women' + "/" + file
    #print(file)
    true_label = None
    if 'gaussian' in file:
        true_label = 'gaussian'
    elif 'poisson' in file:
        true_label = 'poisson'
    if 'mixed' in file:
        true_label = 'mixed'
    # read the image
    image = cv2.imread(file)

    # resize the image
    image = cv2.resize(image, fixed_size)

    # Global Feature extraction
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)

    # Concatenate global features

    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

    # predict label of test image
    prediction = clf.predict(global_feature.reshape(1,-1))[0]
    truelabels.append(true_label)
    predicted_labels.append(train_labels[prediction])
    # show predicted label on image
    #cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    # display the output image
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.savefig('.pdf', format='pdf', dpi=300)
    #plt.savefig(true_label, format='png', dpi=300)

# make predictions on the testing images, finding the index of the
# label with the corresponding largest predicted probability, then
# show a nicely formatted classification report
print("[INFO] evaluating network...")

print(classification_report(truelabels, predicted_labels,
	target_names=train_labels))

#print(confusion_matrix())

cm=confusion_matrix(truelabels, predicted_labels)
print(cm)
y_true = ["Gaussian","Mixed","Poisson"]
#y_true=['Angry', 'Fear', 'Happy','Neutral','Sad']
data = cm
class1_acc = data[0][0]/(data[0][0]+data[0][1]+data[0][2])
class2_acc = data[1][1]/(data[1][0]+data[1][1]+data[1][2])
class3_acc = data[2][2]/(data[2][0]+data[2][1]+data[2][2])
print('Gaussian acc: ',class1_acc)
print('Mixed acc: ',class2_acc)
print('Poisson acc: ',class3_acc)

df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
#df_cm.index.name = 'Actual'
#df_cm.columns.name = 'Predicted'
sn.set(font_scale=2)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 24}, fmt="d")
plt.savefig('Cm_test_ml.pdf', format='pdf', dpi=300)
plt.savefig('Cm_test_ml.png', format='png', dpi=300)
plt.close()




# define our simple neural network
model = Sequential()
model.add(Dense(256, input_shape=(532,), activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="softmax"))
# compile the model
opt = SGD(learning_rate=1e-3, momentum=0.9, decay=1e-3 / 25)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])



# train the network
print("[INFO] training simple network...")
H = model.fit(trainDataGlobal, trainLabelsGlobal, validation_data = (testDataGlobal, testLabelsGlobal),epochs=25)



# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(H.history) 

# save to json:  
hist_json_file = 'history_modeldl.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# or save to csv: 
hist_csv_file = 'history_modeldl.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


# make predictions on the testing images, finding the index of the
# label with the corresponding largest predicted probability, then
# show a nicely formatted classification report
print("[INFO] evaluating network...")
predIdxs = model.predict(testDataGlobal)
print(predIdxs)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testLabelsGlobal, predIdxs,
	target_names=["Gaussian","Mixed","Poisson"]))

#print(confusion_matrix())

cm=confusion_matrix(testLabelsGlobal, predIdxs)
print(cm)
y_true = ["Gaussian","Mixed","Poisson"]
#y_true=['Angry', 'Fear', 'Happy','Neutral','Sad']
data = cm
class1_acc = data[0][0]/(data[0][0]+data[0][1]+data[0][2])
class2_acc = data[1][1]/(data[1][0]+data[1][1]+data[1][2])
class3_acc = data[2][2]/(data[2][0]+data[2][1]+data[2][2])
print('Gaussian acc: ',class1_acc)
print('Mixed acc: ',class2_acc)
print('Poisson acc: ',class3_acc)

df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
#df_cm.index.name = 'Actual'
#df_cm.columns.name = 'Predicted'
sn.set(font_scale=2)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 24}, fmt="d")
plt.savefig('Cm_testml.pdf', format='pdf', dpi=300)
plt.savefig('Cm_testml.png', format='png', dpi=300)
plt.close()




