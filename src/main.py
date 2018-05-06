"""High Level Pipeline for Viola Jones Face Detection."""

from utils.io_tools import read_dataset
from utils.data_tools import preprocess_data
from utils.data_tools import process_data
from utils.data_tools import rescale_data
from utils.integ_img import integral_image
from utils.integ_img import integral_image2
from utils.integ_img import integrate

from violajones.Adaboost import Adaboost

import tensorflow as tf
import numpy as np
import skimage


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
flags.DEFINE_string('valtxtdir', '../data/val.txt',
                    'validation data directory')
flags.DEFINE_string('testtxtdir', '../data/test.txt', 'test data directory')
flags.DEFINE_string('totaltxtdir', '../data/total.txt',
                    'all the image directory')

# Image directory
flags.DEFINE_string('imgdir', '../data/image_data/', 'image data directory')
flags.DEFINE_string('preprocesed_imgdir', '../data/preprocessed_data/',
                    'preprocessed image data directory')
flags.DEFINE_string('rescaled_imgdir', '../data/rescaled_data/',
                    'rescaled image data directory')

###############################################################################
# Paramentes here is for finding bugs ^_^!
###############################################################################


def main(_):
    """High level pipeline."""
    # pp.pprint(flags.FLAGS.__flags)

    # Preprocess method supports ['default', 'rgb', 'hsv']
    preprocess_method = FLAGS.preprocess_method
    feature_type = FLAGS.feature_type

    # Training/Validation/Testing image txt dir
    traintxtdir = FLAGS.traintxtdir
    valtxtdir = FLAGS.valtxtdir
    testtxtdir = FLAGS.testtxtdir
    totaltxtdir = FLAGS.totaltxtdir

    # Training image dataset dir
    imgdir = FLAGS.imgdir
    preprocesed_imgdir = FLAGS.preprocesed_imgdir
    rescaled_imgdir = FLAGS.rescaled_imgdir

    # Read all the images
    raw_dataset, filename = read_dataset(totaltxtdir, imgdir)

    # Resize all the images -> save to new folder, all image in same size
    rescale_data(raw_dataset, filename, feature_type)

    # Preprocess images
    preprocess_data(rescaled_imgdir, preprocesed_imgdir, preprocess_method)

    # Load train/val/test set
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

    # print("[*] Loading val set...")
    # val_set, _ = read_dataset(valtxtdir, preprocesed_imgdir)
    # val_image = val_set['image']
    # val_label = val_set['label']
    # positive_val_img_num = np.sum(val_label == 1)
    # negative_val_img_num = np.sum(val_label == 0)

    # print("[*] Loading test set...")
    # test_set, _ = read_dataset(testtxtdir, preprocesed_imgdir)
    # test_image = test_set['image']
    # test_label = test_set['label']
    # positive_test_img_num = np.sum(test_label == 1)
    # negative_test_img_num = np.sum(test_label == 0)

    # Compute integral image of dataset
    for image in train_image:
        image = integral_image(image)

    # Adaboost and Cascade classifiers
    classifiers = Adaboost(train_image, train_label,
                           positive_train_img_num,
                           negative_train_img_num,
                           feature_size=0)

    print("[*] Loading val set...")
    try:
        val_set, _ = read_dataset(valtxtdir, preprocesed_imgdir)
        val_image = val_set['image']
        val_label = val_set['label']
        positive_val_img_num = np.sum(val_label == 1)
        negative_val_img_num = np.sum(val_label == 0)
    except:
        print("[*] Oops! Please try loading training set again...")
    print("[*] Loading training set successfully!")
    print("[*] " + str(positive_train_img_num) + " faces loaded! " +
          str(negative_train_img_num) + " non-faces loaded!")


    val_set, _ = read_dataset(valtxtdir, preprocesed_imgdir)
    val_image = val_set['image']
    val_label = val_set['label']
    positive_val_img_num = np.sum(val_label == 1)
    negative_val_img_num = np.sum(val_label == 0)

    print('Loading test faces..')
    faces_testing = utils.load_images(pos_testing_path)
    faces_ii_testing = list(map(ii.to_integral_image, faces_testing))
    print('..done. ' + str(len(faces_testing)) + ' faces loaded.\n\nLoading test non faces..')
    non_faces_testing = utils.load_images(neg_testing_path)
    non_faces_ii_testing = list(map(ii.to_integral_image, non_faces_testing))
    print('..done. ' + str(len(non_faces_testing)) + ' non faces loaded.\n')

    print('Testing selected classifiers..')
    correct_faces = 0
    correct_non_faces = 0
    correct_faces = sum(utils.ensemble_vote_all(faces_ii_testing, classifiers))
    correct_non_faces = len(non_faces_testing) - sum(utils.ensemble_vote_all(non_faces_ii_testing, classifiers))

    print('..done.\n\nResult:\n      Faces: ' + str(correct_faces) + '/' + str(len(faces_testing))
          + '  (' + str((float(correct_faces) / len(faces_testing)) * 100) + '%)\n  non-Faces: '
          + str(correct_non_faces) + '/' + str(len(non_faces_testing)) + '  ('
          + str((float(correct_non_faces) / len(non_faces_testing)) * 100) + '%)')

    # Just for fun: putting all haar-like features over each other generates a face-like image
    recon = utils.reconstruct(classifiers, faces_testing[0].shape)
    recon.save('reconstruction.png')



if __name__ == '__main__':
    tf.app.run()
