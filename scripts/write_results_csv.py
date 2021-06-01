import numpy as np
import os
import glob
import tensorflow as tf
import argparse
import random
from tensorflow.keras.preprocessing import image

import ecdl

import csv
import io
import json

parser = argparse.ArgumentParser(description='Choosing the experiment model to load and test')
# parser.add_argument('--exp_id', default='experiments/elbow_efficientnet_lr_1e-4_rot60_ws0.1_hs0.1_zr0.2_round1/weights.16-0.7626.hdf5',type=str,metavar='EXP_ID',help='name of the experiment')
# parser.add_argument('--exp_id', default='experiments/extended_training/weights.25-0.8174.hdf5',type=str,metavar='EXP_ID',help='name of the experiment')
parser.add_argument('--exp_id', default='efficientnetb0_round3',type=str,metavar='EXP_ID',help='name of the experiment')
parser.add_argument('--test_data', default='first',type=str,metavar='EXP_ID',help='Testing Dataset Choice')
parser.add_argument('--load_type', default='model',type=str,metavar='EXP_ID',help='Whether to load model or best weight based on validation result')
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

base_exp_path = 'experiments'
exp_path = os.path.join(base_exp_path, args.exp_id)
args_path = os.path.join(exp_path, 'hyperparameters.json')
with open(args_path) as json_file:
    data = json.load(json_file)

model = tf.keras.models.load_model(exp_path)

if args.load_type == 'weight':
    best_metric = 0.0
    chosen_index = 0
    all_weights = glob.glob(os.path.join(exp_path,'*.hdf5'))
    metric_list = [float(weight_file.split('-')[-1].split('.hdf5')[0]) for weight_file in all_weights]    
    for index,metric_value in enumerate(metric_list):
        if best_metric < metric_value:
            best_metric = metric_value
            chosen_index = index
    model.load_weights(all_weights[chosen_index])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(
        num_thresholds=200, curve='ROC', summation_method='interpolation', name=None,
        dtype=None, thresholds=None, multi_label=False, label_weights=None)
        ])    
# model.summary()

base_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

def evaluate_model(test_dir):
    test_data_gen = base_image_generator.flow_from_directory(batch_size=10,
                                                            directory=test_dir,
                                                            shuffle=False,                                                        
                                                            target_size=(data['image_dims'], data['image_dims']),
                                                            class_mode='categorical',
                                                            )


    results = model.evaluate(test_data_gen)
    # print(results)
    return results

def write_results(test_dir, results):
    csv_file_name = os.path.join('internal_hdd_results', args.exp_id + '_' + os.path.basename(test_dir) + '.csv')
    print(csv_file_name)
    csv_file = io.open(csv_file_name, 'w')
    writer = csv.DictWriter(csv_file, fieldnames=['experiment','image','score'])
    writer.writeheader()
    csv_file.flush()
    
    types = ('*/*.jpg', '*/*.png') # the tuple of file types
    all_files = []
    for files in types:
        all_files.extend(glob.glob(os.path.join(test_dir, files)))
    # all_files = glob.glob(os.path.join(test_dir,'*/*.jpg'))

    for image_path in all_files:
        sample_image_path = image_path

        sample_image = image.load_img(sample_image_path,target_size=(data['image_dims'], data['image_dims']))
        image_array = image.img_to_array(sample_image)

        normalised_image = image_array/255.

        normalised_image = tf.expand_dims(normalised_image,axis=0)

        prediction_result = model.predict(normalised_image)[0][0]

        csv_file = io.open(csv_file_name, 'a')
        writer = csv.DictWriter(csv_file, fieldnames=['experiment','image','score'])
        writer.writerow({'experiment': args.exp_id,
                        'image': sample_image_path,
                        'score': prediction_result})

    ## After all data is filled in                     
    writer.writerow({'experiment': 'Accuracy',
                    'image': 'AUC'})                     
    writer.writerow({'experiment': results[1],
                    'image': results[2]})                 
    csv_file.flush()

if args.test_data == 'first':
    test_dir = 'elbow_data/Clean_Data/ActualTestSet'
    results = evaluate_model(test_dir)
    write_results(test_dir, results)
elif args.test_data == 'second':
    test_dir = 'elbow_data/Clean_Data/SecondTestSet'
    results = evaluate_model(test_dir)
    write_results(test_dir, results)
elif args.test_data == 'both':
    ## Run First
    test_dir = 'elbow_data/Clean_Data/ActualTestSet'
    results = evaluate_model(test_dir)
    write_results(test_dir, results)
    ## Run Second
    test_dir = 'elbow_data/Clean_Data/SecondTestSet'
    results = evaluate_model(test_dir)
    write_results(test_dir, results)
elif args.test_data == 'eval_first':
    ## Run First
    test_dir = 'elbow_data/Clean_Data/ActualTestSet'
    results = evaluate_model(test_dir)
elif args.test_data == 'eval_second':
    ## Run Second
    test_dir = 'elbow_data/Clean_Data/SecondTestSet'
    results = evaluate_model(test_dir)
elif args.test_data == 'eval_both':
    print(args.exp_id)
    test_dir = 'elbow_data/Clean_Data/ActualTestSet'
    results = evaluate_model(test_dir)        
    test_dir = 'elbow_data/Clean_Data/SecondTestSet'
    results = evaluate_model(test_dir)