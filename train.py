# import the necessary packages
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import config
import numpy as np
import pickle
import os
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
#%matplotlib inline



def csv_feature_generator(inputPath, bs, numClasses, mode="train"):
	# open the input file for reading
	f = open(inputPath, "r")
	# loop indefinitely
	while True:
		# initialize our batch of data and labels
		data = []
		labels = []
		# keep looping until we reach our batch size
		while len(data) < bs:
			# attempt to read the next row of the CSV file
			row = f.readline()
            			# check to see if the row is empty, indicating we have
			# reached the end of the file
			if row == "":
				# reset the file pointer to the beginning of the file
				# and re-read the row
				f.seek(0)
				row = f.readline()
				# if we are evaluating we should now break from our
				# loop to ensure we don't continue to fill up the
				# batch from samples at the beginning of the file
				if mode == "eval":
					break
			# extract the class label and features from the row
			row = row.strip().split(",")
			label = row[0]
			label = to_categorical(label, num_classes=numClasses)
			features = np.array(row[1:], dtype="float")
			# update the data and label lists
			data.append(features)
			labels.append(label)
		# yield the batch to the calling function
		yield (np.array(data), np.array(labels))

models = sys.argv[1]
version = sys.argv[2]

# load the label encoder from disk
le = pickle.loads(open('output/le_'+str(models)+'_'+str(version)+'.cpickle', "rb").read())
# derive the paths to the training, validation, and testing CSV files
trainPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(config.TRAIN+str(models)+'_'+str(version))])
valPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(config.VAL+str(models)+'_'+str(version))])
testPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(config.TEST+str(models)+'_'+str(version))])
# determine the total number of images in the training and validation
# sets
totalTrain = sum([1 for l in open(trainPath)])
totalVal = sum([1 for l in open(valPath)])
# extract the testing labels from the CSV file and then determine the
# number of testing images
testLabels = [int(row.split(",")[0]) for row in open(testPath)]
totalTest = len(testLabels)

# construct the training, validation, and testing generators
trainGen = csv_feature_generator(trainPath, config.BATCH_SIZE,
	len(config.CLASSES), mode="train")
valGen = csv_feature_generator(valPath, config.BATCH_SIZE,
	len(config.CLASSES), mode="eval")
testGen = csv_feature_generator(testPath, config.BATCH_SIZE,
	len(config.CLASSES), mode="eval")


# define our simple neural network
model = Sequential()
model.add(Dense(256, input_shape=(7 * 7 * models.output[-1].shape[2],), activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(len(config.CLASSES), activation="softmax"))
# compile the model
opt = SGD(lr=1e-3, momentum=0.9, decay=1e-3 / 25)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])


# train the network
print("[INFO] training simple network...")
H = model.fit(x=trainGen, steps_per_epoch=totalTrain // config.BATCH_SIZE,
	validation_data=valGen, validation_steps=totalVal // config.BATCH_SIZE,epochs=25)



# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(H.history) 

# save to json:  
hist_json_file = 'results/history_model'+str(models)+'_'+str(version)+'.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# or save to csv: 
hist_csv_file = 'results/history_model'+str(models)+'_'+str(version)+'.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


# make predictions on the testing images, finding the index of the
# label with the corresponding largest predicted probability, then
# show a nicely formatted classification report
print("[INFO] evaluating network...")
predIdxs = model.predict(x=testGen,
	steps=(totalTest //config.BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testLabels, predIdxs,
	target_names=le.classes_))

#print(confusion_matrix())

cm=confusion_matrix(testLabels, predIdxs)
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
plt.savefig('results/Cm_'+str(models)+'_'+str(version)+'.pdf', format='pdf', dpi=300)
plt.savefig('results/Cm_'+str(models)+'_'+str(version)+'.png', format='png', dpi=300)


