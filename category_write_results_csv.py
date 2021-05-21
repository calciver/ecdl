import numpy as np
import os
import glob
import tensorflow as tf
import argparse
import random
from tensorflow.keras.preprocessing import image

import utils
import model_files
import csv
import io

parser = argparse.ArgumentParser(description='Choosing the experiment model to load and test')
parser.add_argument('--exp_id', default='experiments/extended_training/weights.25-0.8174.hdf5',type=str,metavar='EXP_ID',help='name of the experiment')
args = parser.parse_args()
print(args)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


model = model_files.tf_enet_model()
model.load_weights(args.exp_id)

model.summary()


csv_file_name = os.path.join('internal_hdd_results',os.path.basename(os.path.dirname(args.exp_id)) + '.csv')

csv_file = io.open(csv_file_name, 'w')
writer = csv.DictWriter(csv_file, fieldnames=['experiment','image','abnormal_score', 'normal_score'])
writer.writeheader()
csv_file.flush()

# After new data was added
# all_files = glob.glob(os.path.join('elbow_data/Clean_Data/ActualTestSet','*/*.jpg'))
all_files = glob.glob(os.path.join('elbow_data/Clean_Data/SecondTestSet','*/*.png'))

for image_path in all_files:
    sample_image_path = image_path

    sample_image = image.load_img(sample_image_path,target_size=(256,256))
    image_array = image.img_to_array(sample_image)

    normalised_image = image_array/255.

    normalised_image = tf.expand_dims(normalised_image,axis=0)

    prediction_result = model.predict(normalised_image)

    csv_file = io.open(csv_file_name, 'a')
    writer = csv.DictWriter(csv_file, fieldnames=['experiment','image','abnormal_score','normal_score'])
    writer.writerow({'experiment': args.exp_id,
                     'image': sample_image_path,
                     'abnormal_score': prediction_result[0, 0],
                     'normal_score': prediction_result[0, 1],
                     })
    csv_file.flush()