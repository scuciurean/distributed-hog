from sklearn.metrics import confusion_matrix
from imageio.v3 import imread
from sklearn import svm
from math import ceil
import progressbar
import numpy as np
import argparse
import pickle
import os

from torch.utils.data import DataLoader

from dataset import grab_datasets, search, load_dataset, group_by_position, hog_dataset
from hog import histogram_of_oriented_gradients
from transform import convert_to_grayscale
from masking import distribute
from filters import Sobel

def prepare_classifier_map(annotated_patches):
    classifiers = []

    active_zones = [idx for idx, data in enumerate(annotated_patches) if len(data) != 0]

    for zone in active_zones:
        data = annotated_patches[zone]
        shapes = np.array([x['shape'] for x in data])
        mean_shape = np.mean(shapes, axis=0).astype(np.int32)
        centers = np.array([x['center'] for x in data])
        mean_position = np.mean(centers, axis=0).astype(np.int32)
        classifiers.append({
            "id" : zone,
            "shape": mean_shape,
            "position": mean_position,
            "coords": (
                (int)(mean_position[0] - (mean_shape[0] / 2)), #xmin
                (int)(mean_position[1] - (mean_shape[1] / 2)), #ymin
                (int)(mean_position[0] + (mean_shape[0] / 2)), #xmax
                (int)(mean_position[1] + (mean_shape[1] / 2)), #ymax
                ),
            "algorithm": svm.SVC()
        })

    return classifiers

def split_dataset(classifier, car_patches, not_car_patches, train_size, batch_size):
    train_data_l0, test_data_l0 = not_car_patches[:int(train_size * len(not_car_patches))], \
                                    not_car_patches[int(train_size * len(not_car_patches)): len(not_car_patches)]
    train_data_l1, test_data_l1 = car_patches[:int(train_size * len(car_patches))], \
                                    car_patches[int(train_size * len(car_patches)): len(car_patches)]

    train_data = train_data_l0 + train_data_l1
    test_data = test_data_l0 + test_data_l1

    def round_to_multiple_of_four(input_tuple):
        rounded_tuple = tuple(round(value / 4) * 4 for value in input_tuple)

        return rounded_tuple

    train_dataset = hog_dataset(train_data, resize_shape=round_to_multiple_of_four(classifier['shape']))
    test_dataset = hog_dataset(test_data, resize_shape=round_to_multiple_of_four(classifier['shape']))

    def custom_collate(batch):
        images, labels = zip(*batch)
        return images, labels

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    return train_loader, test_loader

def prepare_data(classifier_map, annotated_patches):
    annotated_patches = [lp for lp in annotated_patches if lp]
    annotated_labeled_patches = [[(patch['filename'], patch['coords'], 1) for patch in indexed_patches] for indexed_patches in annotated_patches]
    unannotated_labeled_patches = []

    for classifier, car_patches in zip(classifier_map, annotated_patches):
        car_patches_filename = list(set([patch['filename'] for patch in car_patches]))
        patches = []
        for sample_path, _ in dataset:
            if sample_path in car_patches_filename:
                pass
            patches.append((sample_path, classifier['coords'], 0))
        unannotated_labeled_patches.append(patches)

    return annotated_labeled_patches, unannotated_labeled_patches

def extract_features(images, patch_shape=(4,4)):
    image_filter = Sobel()
    features = []
    for image in images:
        _mag, _ang = image_filter.apply(image)

        features.append(histogram_of_oriented_gradients(_mag, _ang, patch_shape, normalize=True).flatten())

    return features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a distributed hog model")

    parser.add_argument('-ts', '--train_size', type=float, help='Model training precentage of the dataset')
    parser.add_argument('-bs', '--batch_size', type=int, help='Number of samples in a batch')

    parser.add_argument('-d', '--name', type=str, help='Dataset name')
    parser.add_argument('-p', '--path', type=str, help='Path for dataset')
    parser.add_argument('-j', '--json', type=str, help='Datasets description file')
    parser.add_argument('-t', '--token', type=str, help='Roboflow access token')
    parser.add_argument('-m', '--mask', type=str, help='Frame mask for the selected dataset')
    parser.add_argument('-c', '--classifier_path', type=str, help='Path for saving trained classifiers')

    args = parser.parse_args()

    datasets_collection = grab_datasets(args.path, args.json, args.token)
    dataset_config = search(datasets_collection, args.name)
    dataset = load_dataset(dataset_config, categories = ['train', 'test'])#['train', 'valid', 'test']

    mask = convert_to_grayscale(imread(args.mask))
    net = distribute(mask, gamma=260, delta=18, slope=15, angle=90, offset=520)*255
    if not os.path.isdir(args.classifier_path):
        os.mkdir(args.classifier_path)
    annotated_patches = group_by_position(dataset, net, exclude=['truck', 'bus', 'undefined'])

    classifier_map = prepare_classifier_map(annotated_patches)
    annotated_labeled_patches, unannotated_labeled_patches = prepare_data(classifier_map, annotated_patches)

    train_size = args.train_size
    batch_size = args.batch_size

    for idx_classifier, (classifier, car_patches, not_car_patches) in enumerate(zip(classifier_map, annotated_labeled_patches, unannotated_labeled_patches)):
        if car_patches < 10 or not_car_patches < 10:
            continue

        train_loader, test_loader = split_dataset(classifier, car_patches, not_car_patches, train_size, batch_size)

        print(f"Training node #{classifier['id']}")
        bar = progressbar.ProgressBar(maxval=len(train_loader),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                      progressbar.Percentage()])
        bar.start()
        X_train, Y_train = [], []
        for idx, (images, labels) in enumerate(train_loader):
            X_train.extend(extract_features(images))
            Y_train.extend(labels)
            bar.update(idx)
        bar.finish()

        local_classifier = classifier["algorithm"]
        local_classifier.fit(np.array(X_train), np.array(Y_train))


        print(f"Testing node #{classifier['id']}")
        bar = progressbar.ProgressBar(maxval=len(test_loader),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                      progressbar.Percentage()])
        bar.start()
        X_test, Y_test = [], []
        for idx, (images, labels) in enumerate(test_loader):
            X_test.extend([local_classifier.predict([f])[0] for f in extract_features(images)])
            Y_test.extend(labels)
            bar.update(idx)
        bar.finish()

        print(confusion_matrix(X_test, Y_test, normalize='true'))
        print(confusion_matrix(X_test, Y_test))
        pickle.dump(local_classifier, open(f"{args.classifier_path}/node{classifier['id']}.svm", 'wb'))