"""Perfoem Modified Adaboost Algorithm in Viola Jones Face Detection."""


import numpy as np
# from functools import partial
# import numpy as np
# from violajones.HaarLikeFeature import HaarLikeFeature
# from violajones.HaarLikeFeature import FeatureTypes
# import progressbar
# from multiprocessing import Pool


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


    :param num_classifiers: Number of classifiers to select, -1 will use all
    classifiers
    :type num_classifiers: int
    :return: List of selected features
    :rtype: list[violajones.HaarLikeFeature.HaarLikeFeature]


    """
    # number of all the images
    num_images = num_pos + num_neg

    # image shape (img_height, img_width)
    img_height, img_width = images[0].shape

    # Maximum feature width and height default to image width and height
    default_size = 24
    feature_height = default_size if feature_size == 0 else img_height
    feature_width = default_size if feature_size == 0 else img_width

    # Create initial weights and labels
    weight_pos = np.ones(num_pos) * 1. / (2 * num_pos)
    weight_neg = np.ones(num_neg) * 1. / (2 * num_neg)
    weights = np.hstack((weight_pos, weight_neg))
    labels = np.hstack((np.ones(num_pos), np.zeros(num_neg)))

    # Create features for all sizes and locations
    pass


def feature_generate():
    """Features generation for Viola Jones Face Detection.

    Generate features for all 5 type, all feature size, all training images

    """
    pass
