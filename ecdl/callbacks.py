import tensorflow as tf
import csv
import os
import ecdl
import glob
import numpy as np

class delete_excess_callback(tf.keras.callbacks.Callback):
  def __init__(self,exp_folder):
    super(delete_excess_callback,self).__init__()    
    ecdl.utils.dir_creator(exp_folder)
    self.exp_folder = exp_folder
    self.best_metric = 0.0

  def on_epoch_end(self,batch,logs=None):
    # batch is basically the epoch number
    all_weights = glob.glob(os.path.join(self.exp_folder,'*.hdf5'))
    delete_status= []
    epoch_list = [float(weight_file.split('-')[-2].split('weights.')[-1]) for weight_file in all_weights]
    metric_list = [float(weight_file.split('-')[-1].split('.hdf5')[0]) for weight_file in all_weights]
    max_epoch = np.max(epoch_list)
    unwanted_indices = []
    for index,metric_value in enumerate(metric_list):
        if self.best_metric < metric_value:
            self.best_metric = metric_value
        else:
            if epoch_list[index]+3 < max_epoch:
                unwanted_indices.append(index)

    for unwanted_index in unwanted_indices:
        os.remove(all_weights[unwanted_index])

    

if __name__ == "__main__":
    all_weights = glob.glob(os.path.join('experiments/elbow_efficientnet_lr_1e-4','*.hdf5'))
    epoch_list = [int(weight_file.split('-')[-2].split('weights.')[-1]) for weight_file in all_weights]
    metric_list = [float(weight_file.split('-')[-1].split('.hdf5')[0]) for weight_file in all_weights]
    print(epoch_list)
    print(metric_list)

    print(np.max(epoch_list))

    