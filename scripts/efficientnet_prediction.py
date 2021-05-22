import numpy as np
import os
import glob
import tensorflow as tf
import argparse
import random
from tensorflow.keras.preprocessing import image

import utils
import model_files


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

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
        
    # train_dir = 'train'

    # base_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split = 0.2)

    # train_data_gen = base_image_generator.flow_from_directory(batch_size=20,
    #                                                         directory=train_dir,
    #                                                         shuffle=True,
    #                                                         class_mode='binary',
    #                                                         subset='training')

    AUC_Metric = tf.keras.metrics.AUC(
        num_thresholds=200, curve='ROC', summation_method='interpolation', name=None,
        dtype=None, thresholds=None, multi_label=False, label_weights=None
        )


    model = model_files.create_efficientnet_model()
    model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        # metrics=[metrics.accuracy_metric,metrics.no_bg_accuracy_metric, metrics.dice_2_metric])
                        metrics=['accuracy',AUC_Metric])



    # Models on Academia Harddrive
    # model.load_weights('experiments/elbow_efficientnet_lr_1e-3_hf_vf_180deg_round1/weights.14-0.7443.hdf5')
    # model.load_weights('experiments/first_try/weights.50-0.925000011920929.hdf5')
    # model.load_weights('experiments/elbow_efficientnet_lr_1e-4_hf_vf_round1/weights.44-0.8311.hdf5')
    # model.load_weights('experiments/elbow_efficientnet_lr_1e-4_rot60_ws0.05_hs0.05_zr0.05_hf_vf_round4/weights.173-0.8311.hdf5')

    # Models on Internal Harddrive
    # model.load_weights('experiments/elbow_efficientnet_lr_1e-4_hf_vf_round1/weights.11-0.6986.hdf5')
    # model.load_weights('experiments/elbow_efficientnet_lr_1e-4_rot60_ws0.1_hs0.1_zr0.2_round1/weights.16-0.7626.hdf5')
    # model.load_weights('experiments/elbow_efficientnet_lr_1e-4_rot60_ws0.05_hs0.05_zr0.05_hf_vf_round1/weights.25-0.7991.hdf5')
    # model.load_weights('experiments/elbow_efficientnet_lr_1e-4_rot60_ws0.05_hs0.05_zr0.05_hf_vf_round3_val_0.1/weights.24-0.8165.hdf5')

    # Models trained on the additional data
    # model.load_weights('experiments/base_effnet_lr_1e-4_rot60_ws0.05_hs0.05_zr0.05_hf_vf_val0.1_round1/weights.7-0.8219.hdf5')
    # model.load_weights('experiments/base_effnet_lr_1e-4_rot60_ws0.05_hs0.05_zr0.05_hf_vf_val0.1_round2/weights.17-0.8904.hdf5')
    # model.load_weights('experiments/base_effnet_lr_1e-4_rot60_ws0.05_hs0.05_zr0.05_hf_vf_val0.1_round3/weights.20-0.8836.hdf5')

    # model.load_weights('experiments/base_effnet_lr_1e-4_rot60_ws0.05_hs0.05_zr0.05_hf_vf_val0.2_round1/weights.25-0.8425.hdf5')
    # model.load_weights('experiments/base_effnet_lr_1e-4_rot60_ws0.05_hs0.05_zr0.05_hf_vf_val0.2_round2/weights.21-0.8390.hdf5')
    # model.load_weights('experiments/base_effnet_lr_1e-4_rot60_ws0.05_hs0.05_zr0.05_hf_vf_val0.2_round3/weights.19-0.8493.hdf5')

    model.load_weights('experiments/new_experiment/weights.18-0.7798.hdf5')

    model.summary()

    # sample_image_path = 'train/cats/cat.2.jpg'
    # sample_image_path = 'train/dogs/dog.2.jpg'

    # sample_image_path = 'SABINE1to3TrainingSetNoQuestionsNoManipulations/Normal/Normal005.jpg'
    # sample_image_path = 'SABINE1to3TrainingSetNoQuestionsNoManipulations/Abnormal/Abnormal010.jpg'

    # sample_image_path = 'SABINE_Test_Set_11Dec19/001_Normal011.jpg'
    # sample_image_path = 'elbow/Classified_Test_Set/Normal/001_Normal011.jpg'

    sample_image_path = 'elbow_data/SABINE 114 size test set/004_Normal022.jpg'

    # sample_image_path = 'elbow/SABINE_114_size_test_set/002_Abnormal001.jpg'

    sample_image = image.load_img(sample_image_path,target_size=(256,256))
    image_array = image.img_to_array(sample_image)

    normalised_image = image_array/255.

    normalised_image = tf.expand_dims(normalised_image,axis=0)

    print(normalised_image.shape)

    print(model(normalised_image))
    # print(model.predict(normalised_image))

    # Basic Test Set
    # test_dir = 'elbow/Classified_Test_Set'

    # Larger Test Set
    test_dir = 'elbow_data/SABINE_Test_Set'

    base_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    test_data_gen = base_image_generator.flow_from_directory(batch_size=10,
                                                                directory=test_dir,
                                                                shuffle=False,
                                                                target_size=(256, 256),
                                                                class_mode='binary',
                                                                )

    # print(test_data_gen[0])
    results = model.evaluate(test_data_gen)

    # print(test_data_gen[0])

    # def predict_image(img_path):
    #     # Read the image and resize it
    #     #img = image.load_img(img_path, target_size=(height, width))
    #     img = image.load_img(img_path)
    #     # Convert it to a Numpy array with target shape.
    #     x = image.img_to_array(img)
    #     # Reshapeow_data/SABINE_Test_Set
    #         animal = "cat"
    #         result = 1 - result
    #     return animal,result

    # cat_images = glob.glob(os.path.join('test','cat*.jpg'))
    # dog_images = glob.glob(os.path.join('test','dog*.jpg'))
    # test_images = glob.glob(os.path.join('test','*.jpg'))


    # cat_image = cat_images[random.randint(0,len(cat_images))]
    # # display.display(Image.open(cat_image))
    # print(predict_image(cat_image))

    # dog_image = dog_images[random.randint(0,len(dog_images))]
    # # display.display(Image.open(dog_image))
    # print(predict_image(dog_image))

    # test_image = test_images[random.randint(0,len(test_images))]
    # # display.display(Image.open(test_image))
    # print(predict_image(test_image))