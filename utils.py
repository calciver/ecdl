import os
import glob
import json

def dir_creator(directory_path):
    if os.path.isdir(directory_path):
        pass
    else:
        os.mkdir(directory_path)
        print(f'The directory {directory_path} does not exist. Creating directory.')

def save_args(args, filename):
    with open(filename, 'w') as fp:
        json.dump(args, fp, default= lambda o: o.__dict__, indent=4)
    return

def read_json(hyperparameters_path):
    with open(hyperparameters_path) as json_file:
        data =json.load(json_file)

    default_dictionary = {'image_dims' : 256}

    for key,value in default_dictionary.items():
        if key not in data:
            data[key] = value

    return data