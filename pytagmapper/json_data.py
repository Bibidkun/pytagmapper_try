import json
import glob
import os
import cv2
import math
import numpy as np
from pytagmapper.geometry import *

def get_path(data_dir, filename):
    return os.path.join(data_dir, filename)

def load_data(data_dir="data"):
    data = {
        'viewpoints': {},
        'camera_matrix': {},
        'tag_side_length': None
    }
    with open(get_path(data_dir, "config.json", 'r')) as f:
        config = json.load(f)
    for camera in config['cameras']:
        camera['camera_matrix'] = np.array(camera['camera_matrix'])
    
    data['camera_matrix'] = config['cameras']
    data['tag_side_length'] = config['tag_side_length']
    for file_path in glob.glob(os.path.join(data_dir, "tags_*.txt")):
        file_name = os.path.splitext(os.path.split(file_path)[-1])[0]
        file_id = file_name.split("_")[-1].strip()
        with open(file_path, "r") as f:
            data['viewpoints'][file_id] = parse_tag_file(f)

    return data