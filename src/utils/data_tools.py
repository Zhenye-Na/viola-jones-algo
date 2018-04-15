"""Image normalization for Viola Jones Face Detection."""

import os
from skimage import io
from skimage.transform import resize
from PIL import Image


def preprocess_data(img_data_dir, img_output_dir, process_method='default'):
    """Preprocesse images.

    Args:
        process_method(str): processing methods needs to support
          ['default', 'rgb', 'hsv'].
        if process_method is 'default':
          1. Convert images to range [0,1].
          2. Convert from rgba to gray-scale then back to rgb. Using skimage.
          (if possible)

        if process_method is 'rgb'
          1. Convert the images to range of [0, 1].
          2. Convert from rgba to rgb. Using skimage. (if possible)

        if process_method is 'hsv':
          1. Convert images to range [0,1].
          2. Convert from rgba to hsv. Using skimage. (if possible)

    Returns:
        Nothing to return.
        Preprocessed image will be automatically saved in file
    """
    # First, we will resize all the images to the same size,
    # which is 24 x 24, the best result given by Viola Jones
    # imgdir = "../data/image_data" = img_data_dir
    # output image dir
    dstdir = "./data/resized_data"

    # If dest directory not exists, create dir
    if not os.path.isdir(dstdir):
        os.mkdir(dstdir)

    # Resize all the images to (24, 24)
    for imname in os.listdir(img_data_dir):
        img = io.imread(os.path.join(img_data_dir, (imname + 'jpg')))
        img = resize(img, (24, 24), mode='reflect')
        io.imsave(os.path.join(dstdir, imname), img)

    # Second, we preprocess all of the image based on
    # selected methods, it may give different accuracy.
    # Change input dir for all of the images.
    inpdir = dstdir

    # If output directory not exists, create dir
    if not os.path.isdir(img_output_dir):
        os.mkdir(img_output_dir)

    if process_method == 'default':
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
                img.convert(mode='RGB')
                img.save(os.path.join(
                    img_output_dir, imname.split('.')[0] + '.jpg'), 'JPEG')

    elif process_method == 'rgb':
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
                img.convert(mode='RGB')
                img.save(os.path.join(
                    img_output_dir, imname.split('.')[0] + '.jpg'), 'JPEG')

    elif process_method == 'hsv':
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
                img.convert(mode='RGB')
                img.save(os.path.join(
                    img_output_dir, imname.split('.')[0] + '.jpg'), 'JPEG')

    else:
        print("Method: " + process_method +
              "is supported here. You wanna give it a try on your own? :)")
