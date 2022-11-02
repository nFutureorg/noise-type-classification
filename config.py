# import the necessary packages
import os
# initialize the path to the *original* input directory of images
#ORIG_INPUT_DATASET = "Food-5K"
# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_PATH = "datasets"
# define the names of the training, testing, and validation
# directories
TRAIN = "train"
TEST = "test"
VAL = "validation"
# initialize the list of class label names
CLASSES = ["gaussian", "poisson","mixed"]
# set the batch size
BATCH_SIZE = 64
# initialize the label encoder file path and the output directory to
# where the extracted features (in CSV file format) will be stored
LE_PATH = os.path.sep.join(["output", "le.cpickle"])
BASE_CSV_PATH = "output"
