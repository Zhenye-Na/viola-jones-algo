"""Perform Modified Adaboost Algorithm in Viola Jones Face Detection."""


import numpy as np
from tqdm import tqdm
from functools import partial
from violajones.HaarFeature import HaarFeature
# from functools import partial
# import numpy as np
# from violajones.HaarLikeFeature import HaarLikeFeature
# from violajones.HaarLikeFeature import FeatureTypes
# import progressbar
# from multiprocessing import Pool


FEATURE_TYPE = {'type-2-y': (1, 2),
                'type-2-x': (2, 1),
                'type-3-y': (3, 1),
                'type-3-x': (1, 3),
                'type-4': (2, 2)}


def AdaBoost(images, groundtruth_labels, num_pos, num_neg, feature_size=0):
    """Perform Adaboost Algorithm in Viola Jones Face Detection.

    Selects a set of classifiers. Iteratively takes the best
    classifiers based on a weighted error.

    Algorithm Overview
        1. Normalize the weights as follows so that w_{i,l} is a probability
            distribution
        2. For each feature j, train a classifier h_j which is restricted to
            using a single feature.
            The classifier’s error rate is evaluated with respect to w_{i,l}
        3. Choose the classifier, h_i, with lowest error ε_i.
        4. Update the weights

    Args:
        images (np.ndarray): Integral training images include positive images
            and negative images. dimension (N * height * width)
        groundtruth_labels (np.ndarray): training images labels, including
            only 0 and 1.
        num_pos (int): number of positive image samples
        num_neg (int): number of negative image samples
        feature_size (int): size of features

    :param num_classifiers: Number of classifiers to select, -1 will use all
    classifiers
    :type num_classifiers: int
    :return: List of selected features
    :rtype: list[violajones.HaarLikeFeature.HaarLikeFeature]


    """
    # ----------------------------------------------------------------------- #
    # Parameters Settings

    # number of all the images
    num_images = num_pos + num_neg

    # image shape (img_height, img_width)
    img_height, img_width = images[0].shape

    # Maximum feature width and height default to image width and height
    default_size = 24
    feature_height = default_size if feature_size == 0 else img_height
    feature_width = default_size if feature_size == 0 else img_width

    # Create initial weights and labels
    weight_pos = np.ones(num_pos).astype(np.float32) / (2 * num_pos)
    weight_neg = np.ones(num_neg).astype(np.float32) / (2 * num_neg)
    weights = np.hstack((weight_pos, weight_neg))
    labels = np.hstack((np.ones(num_pos), np.zeros(num_neg)))

    # ----------------------------------------------------------------------- #
    # Generate features

    # Create features for all sizes and locations
    features = feature_generate(img_height, img_width, feature_height, feature_width)
    num_features = len(features)
    print("[*] Generated " + str(num_features) + "features!")

    votes = np.zeros((num_images, num_features))
    for idx in tqdm(range(num_images)):
        votes[idx, :] = np.array(list(map(partial(get_delta, image=images[idx]), features)))


    # ----------------------------------------------------------------------- #




    # ----------------------------------------------------------------------- #



    # ----------------------------------------------------------------------- #


def feature_generate(img_height, img_width, feature_height, feature_width):
    """Features generation.

    Generate features for all 5 type, all feature size, all training images

    Args:


    Returns:

    """
    features = []
    print("[*] Generate features...")
    for feature in FEATURE_TYPE:
        # default width
        width = FEATURE_TYPE[feature][0]

        for filter_width in tqdm(range(width, feature_width, width)):
            # default height
            height = FEATURE_TYPE[feature][1]

            for filter_height in tqdm(range(height, feature_height, height)):
                for i in range(img_width - filter_width):
                    for j in range(img_height - filter_height):
                        # negative examples
                        features.append(HaarFeature(feature, (i, j),
                                                    filter_width,
                                                    filter_height, 0.05, 0))

                        # positive examples
                        features.append(HaarFeature(feature, (i, j),
                                                    filter_width,
                                                    filter_height, 0.05, 1))

    return features


def get_delta(feature, image):
    """Get delta from those features."""
    return feature._get_delta(image)
