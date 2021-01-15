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
parser.add_argument('--exp_id', default='experiments/elbow_efficientnet_lr_1e-4_rot60_ws0.1_hs0.1_zr0.2_round1/weights.16-0.7626.hdf5',type=str,metavar='EXP_ID',help='name of the experiment')
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


saved_result_path = args.exp_id

# saved_result_path = 'experiments/elbow_efficientnet_lr_1e-4_rot60_ws0.05_hs0.05_zr0.05_hf_vf_round4/weights.173-0.8311.hdf5'

# saved_result_path = 'experiments/elbow_efficientnet_lr_1e-4_hf_vf_round2/weights.21-0.8402.hdf5'

model = model_files.create_efficientnet_model()
model.load_weights(saved_result_path)
model.summary()

csv_file_name = os.path.join('internal_hdd_results',os.path.basename(os.path.dirname(saved_result_path)) + '.csv')
print(csv_file_name)
# csv_file_name = 'results.csv'

csv_file = io.open(csv_file_name, 'w')
writer = csv.DictWriter(csv_file, fieldnames=['experiment','image','score'])
writer.writeheader()
csv_file.flush()

# Before new data
# all_files = glob.glob(os.path.join('elbow/SABINE_114_size_test_set', '*.jpg'))

# After new data was added
all_files = glob.glob(os.path.join('elbow/Full_Processed_Dataset/Test_Set_Full/ActualTestSet','*/*.jpg'))

for image_path in all_files:
    sample_image_path = image_path

    sample_image = image.load_img(sample_image_path,target_size=(256,256))
    image_array = image.img_to_array(sample_image)

    normalised_image = image_array/255.

    normalised_image = tf.expand_dims(normalised_image,axis=0)

    # print(model.predict(normalised_image))

    prediction_result = model.predict(normalised_image)[0][0]

    csv_file = io.open(csv_file_name, 'a')
    writer = csv.DictWriter(csv_file, fieldnames=['experiment','image','score'])
    writer.writerow({'experiment': saved_result_path, 'image': sample_image_path,'score': prediction_result})
    csv_file.flush()