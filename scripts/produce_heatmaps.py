import numpy as np
import os
import glob
import tensorflow as tf
import argparse
import random
import sklearn.metrics
from tensorflow.keras.preprocessing import image
import matplotlib.cm as cm

import ecdl

import csv
import io
import json

import tensorflow_addons as tfa
import matplotlib.pyplot as plt

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

class myargs():
    def __init__(self):
        self.exp_id = 'efficientnetb1_round3'
        self.test_data = 'first'
        self.load_type = 'weight'
   

args = myargs()


base_exp_path = '/workspaces/ecdl/experiments'
exp_path = os.path.join(base_exp_path, args.exp_id)
args_path = os.path.join(exp_path, 'hyperparameters.json')
with open(args_path) as json_file:
    data_json = json.load(json_file)

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


base_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

def produce_model_inputs(img_path, data_json):
    # Read the image and resize it
    img = image.load_img(img_path, target_size=(data_json['image_dims'], data_json['image_dims']))
    # Convert it to a Numpy array with target shape.
    image_array = image.img_to_array(img)
    normalised_image = image_array/255.

    # model_input has shape [1, 256, 256, 3]
    model_input = tf.expand_dims(normalised_image,axis=0)
    return model_input


def predict_image(model_input, model):    
    prediction = model(model_input)    

    if tf.argmax(prediction,axis=-1)[0] == 0:
        result = 'Abnormal'
    elif tf.argmax(prediction,axis=-1)[0] == 1:
        result = 'Normal'

    return prediction, result

def make_gradcam_heatmap(
    img_array,
    model,
    activation_layer,
):
    
    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer(activation_layer)
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(img_array)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature lbow_data/SABINE 114 size test set/004_Normal022.jpgap
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def overlay_heatmap(sample_image_path, heatmap, data_json):
    # We load the original image
    img = tf.keras.preprocessing.image.load_img(sample_image_path, target_size=(data_json['image_dims'], data_json['image_dims']))
    img = tf.keras.preprocessing.image.img_to_array(img)

    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    model_input = produce_model_inputs(sample_image_path, data_json)
    prediction, result = predict_image(model_input,model)

    plt.figure(figsize=[20,20])
    plt.imshow(superimposed_img)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f'Score: {prediction} \nAnswer: {os.path.basename(sample_image_path)}\nPrediction: {result}');


# # sample_image_path = 'elbow_data/SABINE 114 size test set/001_Normal011.jpg'
# # sample_image_path = 'elbow_data/SABINE 114 size test set/002_Abnormal001.jpg'
# # sample_image_path = 'elbow_data/SABINE 114 size test set/003_Abnormal009.jpg'
# # sample_image_path = 'sample_images/elbow_broken.jpeg'
# # sample_image_path = 'sample_images/elbow_fracture_1.jpg'
# sample_image_path = '/workspaces/ecdl/sample_images/elbow_fracture_1.jpg'

# img_array = produce_model_inputs(sample_image_path, data_json)

# # Generate class activation heatmap
# heatmap = make_gradcam_heatmap(
#     img_array,
#     model,
#     activation_layer='top_activation',
# )


# overlay_heatmap(sample_image_path, heatmap, data_json)
# plt.savefig('/workspaces/ecdl/heatmaps/sample.jpg')




types = ('*/*.jpg', '*/*.png') # the tuple of file types
all_files = []
for files in types:
    all_files.extend(glob.glob(os.path.join('/workspaces/ecdl/elbow_data/Clean_Data/SecondTestSet', files)))

for image_path in all_files:
    sample_image_path = image_path

    img_array = produce_model_inputs(sample_image_path, data_json)
    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        activation_layer='top_activation',
    )
    overlay_heatmap(sample_image_path, heatmap, data_json)


    class_path = os.path.dirname(sample_image_path)
    test_path = os.path.dirname(class_path)

    base_name = os.path.basename(sample_image_path)
    real_class = os.path.basename(class_path)
    test_name = os.path.basename(test_path)
    if test_name == 'ActualTestSet':
        saving_base_dir = '/workspaces/ecdl/heatmaps_112'
    elif test_name == 'SecondTestSet':
        saving_base_dir = '/workspaces/ecdl/heatmaps_100'

    if real_class == 'Normal':
        saving_directory = os.path.join(saving_base_dir, 'Normal')
    elif real_class == 'Abnormal':
        saving_directory = os.path.join(saving_base_dir, 'Abnormal')

    final_dir = os.path.join(saving_directory, base_name)

    plt.savefig(final_dir[:-3] + 'jpg')
