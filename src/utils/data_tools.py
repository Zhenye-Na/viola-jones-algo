"""Image normalization for Viola Jones Face Detection."""

import os
import skimage
import numpy as np
from PIL import Image
from skimage import io
from skimage import color
from skimage.transform import resize


def rescale_data(image, filename, feature_type):
    """Resize all the images to the same size.

    Args:
        image (dict): training/validation/test image dataset.

    Returns:
        Nothing to return.
        Rescaled image will be automatically saved in file

    1. Reduced mean (proces images)
    2. Resize all the images to the same size,
    3. Save to new directory
    """
    # data = process_data(image, feature_type)

    dstdir = "../data/image_data/rescaled_data/"

    # If dest directory not exists, create dir
    if not os.path.isdir(dstdir):
        os.mkdir(dstdir)

    data = {}
    data['image'] = []

    for img in image['image']:
        # Resize all the images to same size
        rescaled_img = resize(img, (64, 64), mode='reflect')
        data['image'].append(rescaled_img)

    new_image = process_data(data, feature_type)

    idx = 0
    for img in new_image['image']:
        # Resize all the images to same size
        rescaled_img = resize(img, (64, 64), mode='reflect')
        io.imsave(os.path.join(dstdir, filename[0]), rescaled_img)
        idx += 1


def preprocess_data(img_data_dir, img_output_dir, preprocess_method='default'):
    """Preprocesse images.

    Args:
        process_method(str): processing methods needs to support
          ['default', 'lab', 'hsv'].
        if process_method is 'default':
          1. Convert images to range [0,1].
          2. Convert from rgba to gray-scale.

        if process_method is 'lab'
          1. Convert the images to range of [0, 1].
          2. Convert from rgba to rgb to lab to gray-scale.

        if process_method is 'hsv':
          1. Convert images to range [0,1].
          2. Convert from rgba to hsv to gray-scale.

    Returns:
        Nothing to return.
        Preprocessed image will be automatically saved in file
    """
    inpdir = img_data_dir

    # If output directory not exists, create dir
    if not os.path.isdir(img_output_dir):
        os.mkdir(img_output_dir)

    if preprocess_method == 'default':
        # If the image channel is RGBA, then convert to
        # gray-scale and back to RGB

        for imname in os.listdir(inpdir):

            # Image.open -> opens and identifies the given image file.
            # Return -> an `Image` object.
            img = Image.open(os.path.join(inpdir, imname))

            # Check the original mode (channel for each image).
            if img.mode == 'RGBA':
                # required for img.split()
                img.load()

                # Creates a new image using `L` mode -> gray-scale.
                rgba2gray = Image.new("L", img.size)
                # Convert from gray-scale to RGB.
                gray2rgb = rgba2gray.convert(mode='RGB', colors=256)

                # 3rd is the alpha channel
                gray2rgb.paste(img, mask=img.split()[3])
                gray2rgb.save(os.path.join(
                    img_output_dir, imname.split('.')[0] + '.jpg'), 'JPEG')
            else:
                img.convert(mode='LA')
                img.save(os.path.join(
                    img_output_dir, imname.split('.')[0] + '.jpg'), 'JPEG')

    elif preprocess_method == 'lab':
        # If the image channel is RGBA, then convert to RGB

        for imname in os.listdir(inpdir):
            img = Image.open(os.path.join(inpdir, imname))
            if img.mode == 'RGBA':
                # required for img.split()
                img.load()

                # Creates a new image using `RGB` mode.
                rgba2rgb = Image.new("RGB", img.size)
                rgba2rgb.paste(img, mask=img.split()[3])
                rgba2rgb.save(os.path.join(
                    img_output_dir, imname.split('.')[0] + '.jpg'), 'JPEG')
            else:
                img.convert(mode='LAB')
                img.convert(mode='LA')
                img.save(os.path.join(
                    img_output_dir, imname.split('.')[0] + '.jpg'), 'JPEG')

    elif preprocess_method == 'hsv':
        # If the image channel is RGBA, then convert to hsv

        for imname in os.listdir(inpdir):
            img = Image.open(os.path.join(inpdir, imname))
            if img.mode == 'RGBA':
                # required for img.split()
                img.load()

                # Creates a new image using `HSV` mode.
                rgba2hsv = Image.new("HSV", img.size)
                hsv2rgb = rgba2hsv.convert(mode='RGB', colors=256)
                # 3rd is the alpha channel
                hsv2rgb.paste(img, mask=img.split()[3])
                hsv2rgb.save(os.path.join(
                    img_output_dir, imname.split('.')[0] + '.jpg'), 'JPEG')
            else:
                img.convert(mode='HSV')
                img.convert(mode='LA')
                img.save(os.path.join(
                    img_output_dir, imname.split('.')[0] + '.jpg'), 'JPEG')

    else:
        print("Method: " + preprocess_method +
              "is supported here. You wanna give it a try on your own? :)")


def process_data(data, process_method='default'):
    """Processe dataset.

    Args:
        data(dict): Python dict loaded using io_tools.
        process_method(str): processing methods needs to support
          ['raw', 'default'].

    if process_method is 'raw'
      1. Convert the images to range of [0, 1] by dividing by 255.
      2. Remove dataset mean. Average the images across the batch dimension.
      3. Flatten images, data['image'] is converted to dimension (N, 64*64*3)

    if process_method is 'default':
      1. Convert images to range [0,1]
      2. Convert from rgb to gray then back to rgb. Use skimage
      3. Take the absolute value of the difference with the original image.
      4. Remove dataset mean. Average the absolute value differences across
         the batch dimension. This will result in a mean of dimension (64,64*3).
      5. Flatten images, data['image'] is converted to dimension (N, 64*64*3)

    Returns:
        data(dict): Apply the described processing based on the process_method
        str to data['image'], then return data.
    """
    if process_method == 'raw':
        # Convert images to range [0,1]
        scaled_image = data['image'] / 255
        N = len(scaled_image)

        # Remove dataset mean
        image_mean = np.sum(scaled_image, axis=0) / N
        for image in scaled_image:
            image -= image_mean

        # Flatten images
        data['image'] = scaled_image.flatten().reshape(N, 64 * 64 * 3)

    elif process_method == 'default':
        # Convert images to range [0,1]
        scaled_image = data['image'] / 255
        N = len(scaled_image)

        # Convert from rgb to gray then back to rgb
        grayscale = color.rgb2gray(scaled_image)
        recRgb = color.gray2rgb(grayscale)

        # Take the absolute value of the difference with the original image
        absval = abs(recRgb - scaled_image)

        # Remove dataset mean
        image_mean = np.sum(absval, axis=0) / N

        for image in absval:
            image -= image_mean

        # Flatten images
        data['image'] = absval.flatten().reshape(N, 64 * 64 * 3)

    return data
