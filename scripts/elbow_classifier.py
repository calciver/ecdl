import numpy as np
import os
import glob
import tensorflow as tf
import argparse
import ecdl
import matplotlib.pyplot as plt
import json

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

# tf.debugging.set_log_device_placement(True)

parser = argparse.ArgumentParser(description='Choosing the hyperparameters and saving directory to train an elbow classifier')
parser.add_argument('--exp_id', default='elbow_efficientnet_lr_1e-4',type=str,metavar='EXP_ID',help='name of the experiment')
parser.add_argument('--learning_rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--epochs', default=200, type=int, help='Number of training epochs.')
parser.add_argument('--rotation_range', default=0, type=int, help='Int. Degree range for random rotations.')
parser.add_argument('--width_shift_range', default=0.0, type=float, help='float: fraction of total width, if < 1, or pixels if >= 1')
parser.add_argument('--height_shift_range', default=0.0, type=float, help='float: fraction of total width, if < 1, or pixels if >= 1')
parser.add_argument('--brightness_range', default=None, type=float, nargs=2, help='Tuple or list of two floats. Range for picking a brightness shift value from.')
parser.add_argument('--zoom_range', default=0.0, type=float, help='Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].')
parser.add_argument('--horizontal_flip', default=False, type=bool, help='Boolean. Randomly flip inputs horizontally.')
parser.add_argument('--vertical_flip', default=False, type=bool, help='Boolean. Randomly flip inputs vertically.')
parser.add_argument('--validation_split', default=0.2, type=float, help='Float. Fraction of images reserved for validation (strictly between 0 and 1).')
parser.add_argument('--model', default='efficientnetb3',type=str,metavar='m',help='Model Choice')
parser.add_argument('--model_type', default='b0',type=str,metavar='m',help='More specific model choice')
parser.add_argument('--image_dims', default=256, type=int, help='Image Dimensions to be fed to the model')
parser.add_argument('--class_mode', default='binary',type=str,metavar='m',help='binary/categorical/sparse/input')
parser.add_argument('--batch_size', default=10, type=int, help='Reduced batch size allows greater dimensions')
args = parser.parse_args()
print(args)

base_path = '/workspaces/ecdl'
exp_path = os.path.join(base_path, 'experiments')
model_path = os.path.join(exp_path, args.exp_id)
figure_path = os.path.join(model_path, 'training_loss.png')
history_path = os.path.join(model_path, 'history.json')

ecdl.utils.dir_creator(model_path)

ecdl.utils.save_args(args,os.path.join(model_path,'hyperparameters.json'))

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():        
    # # Larger Dataset
    # train_dir = os.path.join(base_path, 'elbow_data/SABINE_Training_Set')
    train_dir = os.path.join(base_path, 'elbow_data/Large_Training_Set')

    train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=args.rotation_range,
                                                                            width_shift_range=args.width_shift_range,
                                                                            height_shift_range=args.height_shift_range,
                                                                            brightness_range=args.brightness_range,
                                                                            zoom_range=args.zoom_range,
                                                                            horizontal_flip=args.horizontal_flip,
                                                                            vertical_flip=args.vertical_flip,
                                                                            rescale=1./255,
                                                                            validation_split = args.validation_split)

    val_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                          validation_split = args.validation_split)



    train_data_gen = train_image_generator.flow_from_directory(batch_size=args.batch_size,
                                                               target_size=(args.image_dims, args.image_dims),
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               class_mode=args.class_mode,
                                                               subset='training')



    val_data_gen = val_image_generator.flow_from_directory(batch_size=args.batch_size,
                                                           target_size=(args.image_dims, args.image_dims),
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           class_mode=args.class_mode,
                                                           subset='validation')
    if args.model=='efficientnetb0':
        model = ecdl.models.b0_enet_model(output_classes = 2, image_dims=args.image_dims, drop=0.2)
    elif args.model=='efficientnetb1':
        model = ecdl.models.b1_enet_model(output_classes = 2, image_dims=args.image_dims, drop=0.2)        
    elif args.model=='efficientnetb2':
        model = ecdl.models.b2_enet_model(output_classes = 2, image_dims=args.image_dims, drop=0.2)
    elif args.model=='efficientnetb3':
        model = ecdl.models.b3_enet_model(output_classes = 2, image_dims=args.image_dims, drop=0.2)        
    # elif args.model == 'keras_efficientnet':
    #     model = model_files.create_keras_efficientnet_model(relu_units = 120,learning_rate=args.learning_rate)
    # elif args.model == 'keras_xception':
    #     model = model_files.create_efficientnet_model(relu_units = 120,learning_rate=args.learning_rate)
    # elif args.model == 'categorical_efficientnet':
    #     model = model_files.tf_enet_model(output_classes = 2, learning_rate=args.learning_rate, image_dims=args.image_dims)
    # elif args.model == 'b3':
    #     model = model_files.b3_enet_model(output_classes=2, learning_rate=args.learning_rate, image_dims=args.image_dims)
    # else:
    #     print('Model not specified')
    # model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # # Callbacks    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path+'/weights.{epoch}-{val_accuracy:.4f}.hdf5', monitor='val_accuracy', verbose=1,save_best_only=True)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', verbose = 1, patience=50)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(model_path,'result.csv'))
    delete_excess = ecdl.callbacks.delete_excess_callback(exp_folder=model_path)    

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.8,
                                                    patience=5)


    callbacks_list = [checkpoint, earlystopping, csv_logger, delete_excess, reduce_lr]

    hist = model.fit(
        train_data_gen,
        epochs = args.epochs,
        steps_per_epoch = len(train_data_gen),        
        validation_data = val_data_gen,
        verbose = 1,
        validation_steps = len(val_data_gen),
        callbacks=callbacks_list
    )                                                           
    model.save(model_path)

    plt.plot(hist.history['loss'])

    plt.savefig(figure_path)

    history_dictionary = hist.history
    history_dictionary['epoch'] = hist.epoch

    print(history_dictionary)

    # ecdl.utils.save_args(history_dictionary, history_path)

    with open(history_path, 'w') as outfile:
        json.dump(history_dictionary,
                outfile,
                default=lambda o: o.item(),
                # default=lambda o: o.__dict__,
                indent=4)