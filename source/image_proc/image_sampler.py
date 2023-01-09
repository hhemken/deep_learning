#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    module for getting square sample images from existing image files
"""
import numpy
import argparse
import logging
import sys
import json
import argparse
from PIL import Image
import logging
from random import SystemRandom  # does not rely on software state and sequences are not reproducible

import os

__author__ = 'hhemken'
log = logging.getLogger(__file__)
rand = SystemRandom()

RED = 'red'
GREEN = 'green'
BLUE = 'blue'
AVERAGE = 'average'


class ImageSamplerException(Exception):
    """
    Generic exception to be image_sampler within the ImageSampler class.
    """

    def __init__(self, msg):
        msg = 'ImageSamplerException: ' + msg
        log.exception(msg)
        super(ImageSamplerException, self).__init__(msg)


class ImageSampler(object):
    """
    """

    def __init__(self,
                 image_file_path=None,
                 image=None,
                 output_dir='.'):
        """

        :param image_file_path: file system path to incoming image file
        :param output_dir: location where all outputs will be written
        """
        if image_file_path is not None:
            log.info("image_path %s", image_file_path)
            log.info("output_dir %s", output_dir)
            self.image_file_path = image_file_path
            self.image = Image.open(image_file_path)
            self.output_dir = output_dir
            self.image_file_root_name, _ = os.path.splitext(image_file_path)
            self.image_file_root_name = os.path.basename(self.image_file_root_name)
            log.info("image_file_root_name %s", self.image_file_root_name)
        elif image is not None:
            # log.info("received an image object")
            self.image = image

    def get_training_pair_images(self,
                                 ann_input_size,
                                 ann_output_size,
                                 save_images=False,
                                 random_pairs=True,
                                 horizontal_coord=None,
                                 vertical_coord=None):
        """
        
        :return: tuple with input and output image samples
        """
        if ann_input_size >= ann_output_size:
            raise ImageSamplerException('ann_input_size %d is greater than or equal to ann_output_size %d' %
                                        (ann_input_size, ann_output_size))
        # get the output image
        if random_pairs:
            (ann_output_image, sub_box) = get_square_subimage(self.image, int(ann_output_size))
        else:
            (ann_output_image, sub_box) = get_square_subimage(self.image, int(ann_output_size),
                                                              random=False,
                                                              horizontal_coord=horizontal_coord,
                                                              vertical_coord=vertical_coord)
        # get the scaled input image
        ann_input_image = ann_output_image.resize((ann_input_size, ann_input_size), Image.ANTIALIAS)
        if save_images:
            box_string = '%s-%s' % (str(sub_box[0]), str(sub_box[1]))
            ann_input_image_path = '%s/%s-%s-input.png' % (self.output_dir, self.image_file_root_name, box_string)
            ann_output_image_path = '%s/%s-%s-output.png' % (self.output_dir, self.image_file_root_name, box_string)
            ann_input_image.save(ann_input_image_path, quality=100)
            ann_output_image.save(ann_output_image_path, quality=100)
        return ann_input_image, ann_output_image


def get_image_pixel_values(image, desired_channels=[AVERAGE]):
    """

    :param image: The image from which pixel values will be returned, each with a scalar value between 0.0 and 1.0.
    :return: List of pixel values, each a scalar between 00.0 and 1.0. The image is flattened, so that values for
                 line one follow directly after the values of line zero, and so on.
    """
    values = list()
    for pixel in list(image.getdata()):
        red, green, blue, average = flatten_and_rescale_pixel(pixel)
        # first make sure we have all the desired channels
        if red is None and RED in desired_channels:
            values = None
        if green is None and GREEN in desired_channels:
            values = None
        if blue is None and BLUE in desired_channels:
            values = None
        if average is None and AVERAGE in desired_channels:
            values = None
        # if any of the channels were missing, this image doesn't have what we need so we bail and return None
        if values is None:
            break
        # append data to list
        if RED in desired_channels:
            values.append(red)
        if GREEN in desired_channels:
            values.append(green)
        if BLUE in desired_channels:
            values.append(blue)
        if AVERAGE in desired_channels:
            values.append(average)
    return values


def scale_pixel_value(value):
    """

    :param value: scalar value of a single pixel channel, should be between 0 and 255
    :return:
    """
    if value is not None:
        value /= 255.0
        if value < 0.0:
            value = 0.0
        elif value > 1.0:
            value = 1.0
    return value


def flatten_and_rescale_pixel(rgb_tuple, debug=False):
    """

    :param rgb_tuple:
    :param debug:
    :return:
    """
    value = None
    red = None
    green = None
    blue = None
    alpha = None
    average = None
    try:
        try:
            (red, green, blue) = rgb_tuple
            average = (red + green + blue) / 3.0
        except TypeError:
            average = rgb_tuple
        except ValueError:
            (red, green, blue, alpha) = rgb_tuple
            average = (red + green + blue) / 3.0
        if debug:
            log.debug("pixel %s, red %d, green %d, blue %d, average = %d, value = %f, alpha = %s",
                      str(rgb_tuple), str(red), str(green), str(blue), str(average), str(alpha))
    except Exception as except_obj:
        raise ImageSamplerException('rgb_tuple %s: %s' % (rgb_tuple, str(except_obj)))
    return scale_pixel_value(red), scale_pixel_value(green), scale_pixel_value(blue), scale_pixel_value(average)


def get_square_subimage(image, size, random=True, horizontal_coord=None, vertical_coord=None, debug=False):
    """
    Get a square sub-image from the original image with the desired dimension and location, if specified.
    :param debug: whether to log debug-level messages
    :param size: The number of pixels along each side of the square sample.
    :param random: If true (default), the sample will be taken from a random location within the image.
    :param horizontal_coord: If specified, the horizontal coordinate of the sample will have this value.
    :param vertical_coord: If specified, the vertical coordinate of the sample will have this value.
    :return: A PIL Image object containing the sub-image.
    """
    try:
        (width, height) = image.size
        max_horizontal = width - size - 1
        max_vertical = height - size - 1
        if max_horizontal <= 0:
            raise ImageSamplerException('square dimension %d is greater than image width %d' % (size, width))
        if max_vertical <= 0:
            raise ImageSamplerException('square dimension %d is greater than image height %d' % (size, height))
        left_horizontal_coord = horizontal_coord
        upper_vertical_coord = vertical_coord
        if random:
            # randint(a, b) returns a random integer N such that a <= N <= b.
            if horizontal_coord is None:
                left_horizontal_coord = rand.randint(0, max_horizontal)
            if vertical_coord is None:
                upper_vertical_coord = rand.randint(0, max_vertical)
        box = (left_horizontal_coord, upper_vertical_coord,
               left_horizontal_coord + size, upper_vertical_coord + size)
        sub_image = image.crop(box)
        sub_image.load()
    except Exception as except_obj:
        ImageSamplerException('size %s, random %s, horizontal_coord %s, vertical_coord %s: %s' %
                              (size, str(random), str(horizontal_coord), str(vertical_coord), str(except_obj)))
        raise

    return sub_image, box


if __name__ == '__main__':
    log_format = ("%(asctime)s.%(msecs)03d [%(process)d] %(threadName)s: %(levelname)-06s: " +
                  "%(module)s::%(funcName)s:%(lineno)s: %(message)s")
    log_datefmt = "%Y-%m-%dT%H:%M:%S"
    logging.basicConfig(level=logging.DEBUG,
                        format=log_format,
                        datefmt=log_datefmt)

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--input_dir",
                               help="Directory for files to be read in by an operation", action="store")
    parent_parser.add_argument("--output_dir",
                               help="Directory for files to be written by an operation", action="store")
    parent_parser.add_argument("--image_path",
                               help="Input image file", action="store")
    parent_parser.add_argument("--output_path",
                               help="Path for output sub_image file", action="store")
    parent_parser.add_argument("--size",
                               help="Pixel dimension of square sub-image", action="store")
    parent_parser.add_argument("--ann_input_path",
                               help="Output path for ANN input image file", action="store")
    parent_parser.add_argument("--ann_output_path",
                               help="Output path for ANN output image file", action="store")
    parent_parser.add_argument("--ann_input_size",
                               help="Pixel dimension of ANN input square sub-image", action="store")
    parent_parser.add_argument("--ann_output_size",
                               help="Pixel dimension of ANN output square sub-image", action="store")
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest='command')
    # sub_image command
    test_mushroom_parser = subparsers.add_parser('sub_image', parents=[parent_parser],
                                                 help="Extract a square sub-image from an image file.")
    # training_pair command
    test_housing_parser = subparsers.add_parser('training_pair', parents=[parent_parser],
                                                help="Extract a training pair from an image.")

    args = parser.parse_args()
    log.info('args: %s', str(args))

    exit_code = 1
    try:
        if args.command == 'sub_image':
            log.info("running sub_image command")
            # testing static function
            test_image = Image.open(args.image_path)
            (test_sub_image, test_box) = get_square_subimage(test_image, int(args.size))
            log.info("test_box %s", str(test_box))
            test_sub_image.save(args.output_path, quality=100)
        elif args.command == 'training_pair':
            log.info("running training_pair command")
            test_image_sampler = ImageSampler(image_file_path=args.image_path,
                                              output_dir=args.output_dir)
            test_input_img, test_output_img = test_image_sampler.get_training_pair_images(int(args.ann_input_size),
                                                                                          int(args.ann_output_size),
                                                                                          save_images=True)

        exit_code = 0
    except ImageSamplerException as image_sampler_exception:
        log.exception(image_sampler_exception)
    except Exception as generic_exception:
        logging.exception(generic_exception)
    finally:
        logging.shutdown()
        sys.exit(exit_code)
