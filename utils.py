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