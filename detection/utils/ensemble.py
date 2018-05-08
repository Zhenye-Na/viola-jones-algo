"""Perform ensemble method in Viola Jones Face Detection."""

import numpy as np

FEATURE_TYPE = {'type-2-y': (1, 2),
                'type-2-x': (2, 1),
                'type-3-y': (3, 1),
                'type-3-x': (1, 3),
                'type-4': (2, 2)}


def ensemble_vote(int_img, classifiers):
    """Compute vote for integral image using all classifiers.

    Args:
        int_img (np.array): Integral image
        classifiers (list): classifiers selected in AdaBoost

    Returns:
        return 1 iff sum of classifier votes is greater 0, else 0

    """
    vote = sum([classifier._get_delta(int_img) for classifier in classifiers])

    if vote >= 0:
        return 1
    else:
        return 0


def ensemble_vote_all(int_imgs, classifiers):
    """Compute vote for all of the integral images.

    Args:
        int_img (np.array of np.array): Integral images
        classifiers (list): classifiers selected in AdaBoost

    Returns
        return 1 iff sum of classifier votes is greater 0, else 0

    """
    votes = []

    for img in int_imgs:
        votes.append(ensemble_vote(img, classifiers))

    return votes
