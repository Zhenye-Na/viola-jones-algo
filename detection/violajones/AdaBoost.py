"""Perform Modified Adaboost Algorithm in Viola Jones Face Detection."""


import numpy as np
# from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from violajones.HaarFeature import HaarFeature


FEATURE_TYPE = {'type-2-y': (1, 2),
                'type-2-x': (2, 1),
                'type-3-y': (3, 1),
                'type-3-x': (1, 3),
                'type-4': (2, 2)}

# for test
# FEATURE_TYPE = {'type-2-y': (1, 2)}


def AdaBoost(images, labels, num_pos, num_neg, feature_size=0):
    """Perform Adaboost Algorithm in Viola Jones Face Detection.

    Select a set of classifiers using Boosting algorithm.
    Iteratively takes the best classifiers based on a weighted error.

    Algorithm:
        1. Normalize the weights so that w_{i,l} is a probability distribution
        2. For each feature j, train a classifier h_j which is restricted to
           using a single feature.
        3. Choose the best weak classifier with lowest error.
        4. Update the weights.

    Args:
        images (np.ndarray): Integral images include positive images
            and negative images.
        labels (np.ndarray): graound truth labels
        num_pos (int): number of positive image samples
        num_neg (int): number of negative image samples
        feature_size (int): size of features.

    Returns:
        classifiers (list): list of weak classifiers

    """
    # ----------------------------------------------------------------------- #
    # Use all of CPUs for boosting
    pool = Pool(processes=None)

    # Parameters Settings

    # number of all the images
    num_images = num_pos + num_neg

    # image shape
    img_height, img_width = images[0].shape

    # Maximum feature width and height default to image width and height
    default_size = 8
    feature_height = default_size if feature_size == 0 else img_height
    feature_width = default_size if feature_size == 0 else img_width

    # Create initial weights and labels
    weight_pos = np.ones(num_pos).astype(np.float32) / (2 * num_pos)
    weight_neg = np.ones(num_neg).astype(np.float32) / (2 * num_neg)
    weights = np.hstack((weight_pos, weight_neg))

    # ----------------------------------------------------------------------- #
    # Generate features

    # Create features for all sizes and locations
    features = feature_generate(img_height, img_width, feature_height, feature_width)
    num_features = len(features)

    feature_idx = list(range(num_features))

    votes = np.zeros((num_images, num_features))

    print("[*] Getting votes...")
    for idx in range(num_images):
        votes[idx, :] = np.array(list(pool.map(partial(get_delta, image=images[idx]), features))).T
        # print(images[idx].shape)
        if idx % 5 == 0:
            print("[*] Got " + str(idx) + " votes now!")

    print("[*] Generated " + str(num_features) + " features!")
    # ----------------------------------------------------------------------- #
    # Adboost algorithm  - selecting classifiers

    classifiers = []

    print("[*] Selecting " + str(15) + " classifiers...")
    for i in range(15):

        # Re-intialize prediction in every iteration
        pred = np.zeros(len(feature_idx))

        # Normalize weight in each iteration
        weights *= 1. / np.sum(weights)

        for j in range(len(feature_idx)):
            idx = feature_idx[j]
            error = sum(map(lambda img_idx: weights[img_idx] if labels[img_idx] != votes[img_idx, idx] else 0, range(num_images)))
            pred[j] = error

    # ----------------------------------------------------------------------- #
        # Select best weak classifer
        min_error_idx = np.argmin(pred)
        best_error = pred[min_error_idx]
        best_feature_idx = feature_idx[min_error_idx]

        # Re-set feature weights
        classifier = features[best_feature_idx]
        feature_weight = 0.5 * np.log((1 - best_error) / best_error)
        classifier.weight = feature_weight

        classifiers.append(classifier)

    # ----------------------------------------------------------------------- #
        # Update weights
        weights = np.array(list(map(lambda img_idx: weights[img_idx] * np.sqrt((1 - best_error) / best_error) if labels[img_idx] != votes[img_idx, best_feature_idx] else weights[img_idx] * np.sqrt(best_error / (1 - best_error)), range(num_images))))

        # Remove selected classifier
        feature_idx.remove(best_feature_idx)

        if i % 5 == 0:
            print("[*] Slected " + str(i) + " classifiers now!")

    return classifiers


def feature_generate(img_height, img_width, feature_height, feature_width):
    """Generate features.

    Generate features for all 5 type, all feature size, all training images

    Args:
        img_height (int): height of integral image
        img_width (int): width of integral image
        feature_height (int): height of feature window (upper-bound)
        feature_width (int): width of feature window (upper-bound)

    Returns:
        features (list): list of features generated

    """
    features = []
    print("[*] Generating features...")
    for feature in FEATURE_TYPE:
        # default width
        width = FEATURE_TYPE[feature][1]

        # Start with width for all features or with any number
        for filter_width in range(5, feature_width, width):
            # default height
            height = FEATURE_TYPE[feature][0]

            # Start with height for all features or with any number
            for filter_height in range(5, feature_height, height):

                for i in range(img_width - filter_width):
                    for j in range(img_height - filter_height):

                        # negative examples
                        features.append(HaarFeature(feature, (i, j),
                                                    filter_width,
                                                    filter_height, 0.01, 0))

                        # positive examples
                        features.append(HaarFeature(feature, (i, j),
                                                    filter_width,
                                                    filter_height, 0.01, 1))

    return features


def get_delta(feature, image):
    """Get delta from those features.

    Args:
        feature (HaarFeature): feature
        image (np.ndarray): integral image

    Return:
        delta

    """
    return feature._get_delta(image)
