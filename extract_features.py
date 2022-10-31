# import the necessary packages
import sys
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50, Xception, VGG16, InceptionV3, DenseNet121, MobileNetV2, EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7


from tensorflow.keras.applications.resnet50 import preprocess_input as rs
from tensorflow.keras.applications.xception import preprocess_input as xc

from tensorflow.keras.applications.vgg16 import preprocess_input as vg
from tensorflow.keras.applications.inception_v3 import preprocess_input as incept

from tensorflow.keras.applications.densenet import preprocess_input as dsnet
from tensorflow.keras.applications.mobilenet import preprocess_input as mbnet

from tensorflow.keras.applications.efficientnet import preprocess_input as efnet


from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import config
from imutils import paths
import numpy as np
import pickle
import random
import os


#[]
#version = 0
models = sys.argv[1]
version = sys.argv[2]

# load the ResNet50 network and initialize the label encoder
print("[INFO] loading network...")

if models == 'resnet':
    model = ResNet50(weights="imagenet", include_top=False)
    processing_input = rs
    width = 224
    height = 224
    out = model.output[-1]
elif models == 'vgg':
    model = VGG16(weights="imagenet", include_top=False)
    processing_input = vg
    width = 224
    height = 224
    out = model.output[-1].shape[2]
elif models == 'xception':
    model = Xception(weights="imagenet", include_top=False)
    processing_input = xc
    width = 299
    height = 299
    out = model.output[-1].shape[2]
elif models == 'inception':
    model = InceptionV3(weights="imagenet", include_top=False)   
    processing_input = incept
    width = 299
    height = 299
    out = model.output[-1].shape[2]
elif models == 'densenet':
    model = DenseNet121(weights="imagenet", include_top=False)
    processing_input = dsnet
    width = 224
    height = 224
    out = model.output[-1].shape[2]
elif models == 'mobilenet':
    model = MobileNetV2(weights="imagenet", include_top=False)
    processing_input = mbnet
    width = 224
    height = 224
    out = model.output[-1].shape[2]
elif models == 'efficientnet' and version :
    processing_input = efnet
    if version == 0:
        model = EfficientNetB0(weights="imagenet", include_top=False)
        width = 224
        height = 224
        out = model.output[-1].shape[2]
    elif version == 1:
        model = EfficientNetB1(weights="imagenet", include_top=False) 
        width = 240
        height = 240
        out = model.output[-1].shape[2]
    elif version == 2:
        model = EfficientNetB2(weights="imagenet", include_top=False)  
        width = 260
        height = 260
        out = model.output[-1].shape[2]
    elif version == 3:
        model = EfficientNetB3(weights="imagenet", include_top=False)  
        width = 300
        height = 300
        out = model.output[-1].shape[2]
    elif version == 4:
        model = EfficientNetB4(weights="imagenet", include_top=False)  
        width = 380
        height = 380
        out = model.output[-1].shape[2]
    elif version == 5:
        model = EfficientNetB5(weights="imagenet", include_top=False)
        width = 456
        height = 456
        out = model.output[-1].shape[2]
    elif version == 6:
        model = EfficientNetB6(weights="imagenet", include_top=False)
        width = 528
        height = 528
        out = model.output[-1].shape[2]
    else:
        model = EfficientNetB7(weights="imagenet", include_top=False) 
        width = 600
        height = 600
        out = model.output[-1].shape[2]
else:
    print('Input the name of the model')

le = None



# loop over the data splits
for split in (config.TRAIN, config.TEST, config.VAL):
    # grab all image paths in the current split
    print("[INFO] processing '{} split'...".format(split))
    p = os.path.sep.join([config.BASE_PATH, split])
    imagePaths = list(paths.list_images(p))
    # randomly shuffle the image paths and then extract the class
    # labels from the file paths
    random.shuffle(imagePaths)
    labels = [p.split(os.path.sep)[-2] for p in imagePaths]
    # if the label encoder is None, create it
    if le is None:
        le = LabelEncoder()
        le.fit(labels)
    # open the output CSV file for writing
    csvPath = os.path.sep.join([config.BASE_CSV_PATH,
        "{}.csv".format(split+str(models)+'_'+str(version))])
    csv = open(csvPath, "w")
    # loop over the images in batches
    for (b, i) in enumerate(range(0, len(imagePaths), config.BATCH_SIZE)):
        # extract the batch of images and labels, then initialize the
        # list of actual images that will be passed through the network
        # for feature extraction
        print("[INFO] processing batch {}/{}".format(b + 1,
            int(np.ceil(len(imagePaths) / float(config.BATCH_SIZE)))))
        batchPaths = imagePaths[i:i + config.BATCH_SIZE]
        batchLabels = le.transform(labels[i:i + config.BATCH_SIZE])
        batchImages = []
        #glob_features = []
        # loop over the images and labels in the current batch
        for imagePath in batchPaths:
            # load the input image using the Keras helper utility
            # while ensuring the image is resized to 224x224 pixels
            image = load_img(imagePath, target_size=(width, height))
            image = img_to_array(image)
            # preprocess the image by (1) expanding the dimensions and
            # (2) subtracting the mean RGB pixel intensity from the
            # ImageNet dataset
            image = np.expand_dims(image, axis=0)
            image = processing_input(image)
            # add the image to the batch
            #print(batchImages)
            batchImages.append(batchImages)
            #glob_features.append(features)
            #print(batchImages)
                    # pass the images through the network and use the outputs as
            # our actual features, then reshape the features into a
            # flattened volume
        batchImages = np.vstack(batchImages)
        features = model.predict(batchImages, batch_size=config.BATCH_SIZE)
        features = features.reshape((features.shape[0], 7 * 7 * model.output[-1].shape[2]))
        # loop over the class labels and extracted features
        for (label, vec) in zip(batchLabels, features):
            # construct a row that exists of the class label and
            # extracted features
            vec = ",".join([str(v) for v in vec])
            csv.write("{},{}\n".format(label, vec))
    # close the CSV file
    csv.close()
# serialize the label encoder to disk
f = open('output/le_'+str(models)+'_'+str(version)+'.cpickle', "wb")
f.write(pickle.dumps(le))
f.close()
