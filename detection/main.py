"""High Level Pipeline for Viola Jones Face Detection."""

from utils.io_tools import read_dataset
from utils.data_tools import preprocess_data
from utils.data_tools import rescale_data
from utils.integ_img import integral_image
from utils.ensemble import ensemble_vote_all
from violajones.AdaBoost import AdaBoost

import tensorflow as tf
import numpy as np


flags = tf.app.flags
FLAGS = flags.FLAGS

# Define preprocess method
flags.DEFINE_string(
    'preprocess_method',
    'default',
    'Feature type, supports [default, lab, hsv]')

# Define process method (reduced mean)
flags.DEFINE_string(
    'feature_type',
    'default',
    'Feature type, supports [default, raw]')

# Training/Validation/Testing txt dir
flags.DEFINE_string('traintxtdir', '../data/train.txt',
                    'training data directory')
# flags.DEFINE_string('valtxtdir', '../data/val.txt',
#                     'validation data directory')
flags.DEFINE_string('testtxtdir', '../data/test.txt', 'test data directory')
flags.DEFINE_string('totaltxtdir', '../data/total.txt',
                    'all the image directory')

# Image directory
flags.DEFINE_string('imgdir', '../data/image_data/', 'image data directory')
flags.DEFINE_string('rescaled_imgdir', '../data/rescaled_data/',
                    'rescaled image data directory')
flags.DEFINE_string('preprocesed_imgdir', '../data/preprocessed_data/',
                    'preprocessed image data directory')


def main(_):
    """High level pipeline."""
    # pp.pprint(flags.FLAGS.__flags)

    # Preprocess method supports ['default', 'rgb', 'hsv']
    preprocess_method = FLAGS.preprocess_method
    feature_type = FLAGS.feature_type

    # Training/Validation/Testing image txt dir
    traintxtdir = FLAGS.traintxtdir
    # valtxtdir = FLAGS.valtxtdir
    testtxtdir = FLAGS.testtxtdir

    # All the image together
    totaltxtdir = FLAGS.totaltxtdir

    # Training image dataset dir
    imgdir = FLAGS.imgdir
    preprocesed_imgdir = FLAGS.preprocesed_imgdir
    rescaled_imgdir = FLAGS.rescaled_imgdir

    # Read all the images
    # -------------------------------------------------------------- #
    # print("[*] Reading and preprocessing dataset...")
    # raw_dataset, filename = read_dataset(totaltxtdir, imgdir)

    # Resize all the images -> save to new folder, all image in same size
    # rescale_data(raw_dataset, filename, feature_type)

    # Preprocess images
    # preprocess_data(rescaled_imgdir, preprocesed_imgdir, preprocess_method)

    # Load train/val/test set
    # -------------------------------------------------------------- #
    print("[*] Loading training set...")
    try:
        train_set, _ = read_dataset(traintxtdir, preprocesed_imgdir)
        train_image = train_set['image']
        train_label = train_set['label']
        positive_train_img_num = np.sum(train_label == 1)
        negative_train_img_num = np.sum(train_label == 0)
    except:
        print("[*] Oops! Please try loading training set again...")

    print("[*] Loading training set successfully!")
    print("[*] " + str(positive_train_img_num) + " faces loaded! " +
          str(negative_train_img_num) + " non-faces loaded!")

    # Compute integral image of training set
    for idx in range(train_image.shape[0]):
        tmp = integral_image(train_image[idx])
        train_image[idx] = tmp

    # Adaboost and Cascade classifiers
    classifiers = AdaBoost(train_image, train_label,
                           positive_train_img_num,
                           negative_train_img_num,
                           feature_size=0)

    # -------------------------------------------------------------- #
    print("[*] Loading test set...")
    try:
        test_set, _ = read_dataset(testtxtdir, preprocesed_imgdir)
        test_image = test_set['image']
        test_label = test_set['label']
        positive_test_img_num = np.sum(test_label == 1)
        negative_test_img_num = np.sum(test_label == 0)
    except:
        print("[*] Oops! Please try loading test set again...")

    print("[*] Loading test set successfully!")
    print("[*] " + str(positive_test_img_num) + " faces loaded! " +
          str(negative_test_img_num) + " non-faces loaded!")

    # Compute integral image of test set
    for idx in range(test_image.shape[0]):
        tmp = integral_image(test_image[idx])
        test_image[idx] = tmp

    # Start test
    # -------------------------------------------------------------- #
    print("[*] Start testing...")

    positive_test_images = []
    negative_test_images = []

    for idx in range(positive_test_img_num + negative_test_img_num):
        if test_label[idx]:
            positive_test_images.append(test_image[idx])
        else:
            negative_test_images.append(test_image[idx])

    pred_pos = np.sum(ensemble_vote_all(np.array(positive_test_images), classifiers))
    acc_pos = float(pred_pos / positive_test_img_num)

    pred_neg = np.sum(ensemble_vote_all(np.array(negative_test_images), classifiers))
    acc_neg = float(pred_pos / positive_test_img_num)

    print("[*] Test done!")
    print("Faces [" + str(pred_pos) + " / " +
          str(positive_test_img_num) + "] accuracy: " + str(acc_pos))
    print("Objects [" + str(pred_neg) + " / " +
          str(negative_test_img_num) + "] accuracy: " + str(acc_neg))


if __name__ == '__main__':
    tf.app.run()
