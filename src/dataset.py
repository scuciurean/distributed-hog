from skimage.transform import resize
from checksumdir import dirhash
from itertools import groupby
from roboflow import Roboflow
from imageio.v3 import imread
import pandas as pd
import numpy as np
import json
import math
import os

from torch.utils.data import Dataset

from transform import convert_to_grayscale

class hog_dataset(Dataset):
    def __init__(self, data, resize_shape=None):
        self.data = data
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, coords, label = self.data[idx]
        xmin, ymin, xmax, ymax = coords
        image = imread(filename)
        image = convert_to_grayscale(image)
        patch = image[ymin:ymax, xmin:xmax]
        if self.resize_shape:
            patch = resize(patch, self.resize_shape, anti_aliasing=True)

        return patch, label

def grab_datasets(destination_path, metadata, token):
    with open(metadata) as f:
        dataset_config = json.load(f)
        if not os.path.isdir(destination_path): os.mkdir(destination_path)

    rf = Roboflow(api_key=token)

    for d in dataset_config['datasets']:
        path = f"{destination_path}/{d['user']}-v{d['version']}"
        d['path'] = path
        if os.path.isdir(path) and dirhash(path, excluded_extensions=['.txt']) == d['checksum']:
            continue
        rf.workspace(d['user']).project(d['project']).version(d['version']).download(
            model_format=dataset_config['format'],
            location=path,
            overwrite=True)

    return dataset_config

def validate_dataset(values):
    for _, group in values:
        for element in group:
            _, _, _, xmin, ymin, xmax, ymax = element
            if xmin > xmax or ymin > ymax:
                group.remove(element)

def load_dataset(dataset, categories):
    annotations = []
    for item in categories:
        subdir = f'{dataset["path"]}/{item}/'
        if os.path.isdir(subdir):
            data = pd.read_csv(f"{subdir}/_annotations.csv")
            data['filename'] = subdir + data['filename']
            annotations.append(data)

    annotations = pd.concat(annotations, ignore_index=True).sort_values(by='filename')

    grouped_values = []
    for key, group in groupby(annotations.values, key=lambda x: x[0]):
        values = [item[1:] for item in group]
        grouped_values.append((key, values))

    validate_dataset(grouped_values)

    return grouped_values

def search(datsaets, project):
    df = pd.DataFrame(datsaets['datasets'])
    result = df.query(f"user == '{project}'").to_dict(orient='records')

    return result[0]

def group_by_position(dataset, net, exclude):
    net_list = list(zip(np.nonzero(net)[0], np.nonzero(net)[1]))
    grouped_patches = [[] for _ in range(len(net_list))]

    for filename, annotations in dataset:
        for _, _, dclass, xmin, ymin, xmax, ymax in annotations:
            if dclass in exclude:
                continue
            for idx, pos in enumerate(net_list):
                x, y = pos
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    grouped_patches[idx].append({
                        "class": dclass,
                        "filename": filename,
                        "shape": (xmax-xmin, ymax-ymin),
                        "coords": (xmin, ymin, xmax, ymax),
                        "center": ((int)(xmax + (xmax - xmin) / 2), (int)(ymax + (ymax - ymin) / 2))
                    })

    return grouped_patches