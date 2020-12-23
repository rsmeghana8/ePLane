import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# Keras shit
from keras.preprocessing.image import load_img,img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
gpuconfig = tf.ConfigProto()
gpuconfig.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=gpuconfig))


# define 81 classes that the coco model knowns about

category = 'isaid'
class_names = ['storage_tank', 'Large_Vehicle', 'Small_Vehicle', 'plane',
                'ship', 'Swimming_pool', 'Harbor', 'tennis_court',
                'Ground_Track_Field', 'Soccer_ball_field', 'baseball_diamond',
                'Bridge', 'basketball_court', 'Roundabout', 'Helicopter']


#class_names = ['bread-wholemeal', 'potatoes-steamed', 'broccoli', 'butter', 
            # 'hard-cheese', 'water', 'banana', 'wine-white', 'bread-white', 
            # 'apple', 'pizza-margherita-baked', 'salad-leaf-salad-green', 
            # 'zucchini', 'water-mineral', 'coffee-with-caffeine', 'avocado', 
            # 'tomato', 'dark-chocolate', 'white-coffee-with-caffeine', 'egg', 
            # 'mixed-salad-chopped-without-sauce', 'sweet-pepper', 'mixed-vegetables', 
            # 'mayonnaise', 'rice', 'chips-french-fries', 'carrot', 'tomato-sauce', 
            # 'cucumber', 'wine-red', 'cheese', 'strawberries', 'espresso-with-caffeine', 
            # 'tea', 'chicken', 'jam', 'leaf-spinach', 'pasta-spaghetti', 'french-beans', 'bread-whole-wheat']

# define the test configuration
class TestConfig(Config):
    NAME = 'food'

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(class_names)  # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.1

def main():
    array = sys.argv[1:]

    if os.path.exists(array[0]):
        path_to_weight = array[0]
        print(path_to_weight)
    else:
        print('path to weight does not exist')
        sys.exit(0)

    if os.path.exists(array[1]):
        path_to_image =array[1]
        print(path_to_image)
    else:
        print('path to image does not exist')
        sys.exit(0)

    if float(array[2]) <= 1 and float(array[2]) >= 0:
        conf=float(array[2])
    else:
        print('confidence must be a float')
        sys.exit(0)

    config = TestConfig()
    config.DETECTION_MIN_CONFIDENCE = conf

	# define the model
    rcnn = MaskRCNN(mode='inference', model_dir='./load_weights', config=config)
	# load coco model weights
    rcnn.load_weights(path_to_weight, by_name=True)
	# load photograph
    img = cv2.imread(path_to_image)
    dim = (800,800)
    img = cv2.resize(img,dim,interpolation = cv2.INTER_AREA)
    img = img_to_array(img)
    
	# make prediction
    results = rcnn.detect([img], verbose=1)
	# get dictionary for first prediction
    r = results[0]
    # print('Result:\n', r)
	# show photo with bounding boxes, masks, class labels and scores
    display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, 
                        r['scores'])
    print('Done displaying')
if __name__ == '__main__':
	main()

