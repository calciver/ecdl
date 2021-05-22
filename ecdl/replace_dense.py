import tensorflow as tf

model_builder = tf.keras.applications.efficientnet.EfficientNetB0
# img_size = (256, 256)
# preprocess_input = tf.keras.applications.xception.preprocess_input
# decode_predictions = tf.keras.applications.xception.decode_predictions

# last_conv_layer_name = "block14_sepconv2_act"
# classifier_layer_names = [
#     "avg_pool",
#     "predictions",
# ]

efficientnet_model_base = model_builder(weights="imagenet",include_top=True, pooling='max')
efficientnet_model_base.summary()

# efficientnet_model_base.predictions = tf.keras.layers.Dense(units=2)
# efficientnet_model_base.predictions.build([None, 1280])

efficientnet_model_base.layers.pop()
efficientnet_model_base.add(tf.keras.layers.Dense(units=2))
# set_layer('predictions') = tf.keras.layers.Dense(units=2)
# model.get_layer(layer_name)(x)



efficientnet_model_base.summary()
