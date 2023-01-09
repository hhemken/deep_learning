#!/usr/bin/python
"""
    Module with tools to get sample sub-images from a bitmap image.
"""
import argparse
from PIL import Image
import logging
from random import SystemRandom  # does not rely on software state and sequences are not reproducible

import os

# import stat
# import re

log = logging.getLogger()


def get_square_subimage(image, size, random=True, horizontal_coord=None, vertical_coord=None, debug=False):
    """
    Get a square sub-image from the original image with the desired dimension and location, if specified.
    :param debug: whether to log debug-level messages
    :param image: image to be sampled
    :type image: PIL.image
    :param size: The number of pixels along each side of the square sample.
    :param random: If true (default), the sample will be taken from a random location within the image.
    :param horizontal_coord: If specified, the horizontal coordinat of the sample will have this value.
    :param vertical_coord: If specified, the vertical coordinat of the sample will have this value.
    :return: A PIL Image object containing the sub-image.
    """
    try:
        rand = SystemRandom()
        left_horizontal_coord = horizontal_coord
        upper_vertical_coord = vertical_coord
        (width, height) = image.size
        max_horizontal = width - size - 1
        max_vertical = height - size - 1
        if max_horizontal <= 0:
            raise Exception('square dimension %d is greater than image width %d' % (size, width))
        if max_vertical <= 0:
            raise Exception('square dimension %d is greater than image height %d' % (size, height))
        if random:
            # randint(a, b) returns a random integer N such that a <= N <= b.
            if horizontal_coord is None:
                left_horizontal_coord = rand.randint(0, max_horizontal)
            if vertical_coord is None:
                upper_vertical_coord = rand.randint(0, max_vertical)
        # im.crop(box) => image
        # Returns a rectangular region from the current image. The box is a 4-tuple defining
        #   the left, upper, right, and lower pixel coordinate.
        # This is a lazy operation. Changes to the source image may or may not be reflected in the cropped image.
        # To get a separate copy, call the load method on the cropped copy.
        box = (left_horizontal_coord, upper_vertical_coord,
               left_horizontal_coord + size, upper_vertical_coord + size)
        sub_image = image.crop(box)
        sub_image.load()
    except Exception as except_obj:
        log.exception('size %s, random %s, horizontal_coord %s, vertical_coord %s: %s',
                      size, str(random), str(horizontal_coord), str(vertical_coord), str(except_obj))
        raise

    return sub_image, box


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
            value = average / 255.0
        except TypeError:
            value = rgb_tuple / 255.0
        except ValueError:
            (red, green, blue, alpha) = rgb_tuple
            average = (red + green + blue) / 3.0
            value = average / 255.0
        if debug:
            log.debug("pixel %s, red %d, green %d, blue %d, average = %d, value = %f, alpha = %s",
                      str(rgb_tuple), red, green, blue, average, value, str(alpha))
    except Exception as except_obj:
        log.exception('rgb_tuple %s: %s', rgb_tuple, str(except_obj))
        raise
    if value < 0.0:
        value = 0.0
    elif value > 1.0:
        value = 1.0
    return value


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    log.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd', help='Sub-commands')
    # create the parser for the "sub_image" command
    parser_sub_image = subparsers.add_parser('sub_image', help='Extract a square sub-image from an image file')
    parser_sub_image.add_argument("--image_path", "-i", required=True,
                                  help="Input image file", action="store")
    parser_sub_image.add_argument("--output_path", "-o", required=True,
                                  help="Path for output sub_image file", action="store")
    parser_sub_image.add_argument("--size", "-s", required=True,
                                  help="Pixel dimension of square sub-image", action="store")
    # create the parser for the "training_pair" command
    parser_sub_image = subparsers.add_parser('training_pair', help='Extract a square sub-image from an image file')
    parser_sub_image.add_argument("--image_path", "-p", required=True,
                                  help="Input image file", action="store")
    parser_sub_image.add_argument("--ann_input_path", "-i", required=True,
                                  help="Output path for ANN input image file", action="store")
    parser_sub_image.add_argument("--ann_output_path", "-o", required=True,
                                  help="Output path for ANN output image file", action="store")
    parser_sub_image.add_argument("--ann_input_size", "-m", required=True,
                                  help="Pixel dimension of ANN input square sub-image", action="store")
    parser_sub_image.add_argument("--ann_output_size", "-n", required=True,
                                  help="Pixel dimension of ANN output square sub-image", action="store")

    options = parser.parse_args()

    if options.cmd == 'sub_image':
        log.info("image_path %s", options.image_path)
        test_image = Image.open(options.image_path)
        (test_sub_image, test_box) = get_square_subimage(test_image, int(options.size))
        log.info("test_box %s", str(test_box))
        test_sub_image.save(options.output_path, quality=100)

    elif options.cmd == 'training_pair':
        log.info("image_path %s", options.image_path)
        test_image = Image.open(options.image_path)
        (ann_input, test_box) = get_square_subimage(test_image, int(options.ann_input_size))
        log.info("ann_input test_box %s", str(test_box))
        (ann_output, test_box) = get_square_subimage(ann_input, int(options.ann_output_size), random=False,
                                                     horizontal_coord=int(options.ann_output_size),
                                                     vertical_coord=int(options.ann_output_size))
        ann_output.save(options.ann_output_path, quality=100)
        scaled_width = 2 * (int(options.ann_input_size) // 3)
        scaled_height = 2 * (int(options.ann_input_size) // 3)
        ann_input = ann_input.resize((scaled_width, scaled_height), Image.ANTIALIAS)
        ann_input.save(options.ann_input_path, quality=100)
        test_image_bands = test_image.getbands()
        log.info("image bands: %s", str(test_image_bands))
        test_image_band_sub_images = list(test_image.split())
        input_head, input_tail = os.path.split(options.image_path)
        output_head, output_tail = os.path.split(options.ann_input_path)
        for index in range(0, len(test_image_band_sub_images)):
            if test_image_bands[index] == 'A':
                continue
            test_band_file_name = '%s/%s-band_%s.jpg' % (output_head,
                                                         os.path.splitext(input_tail)[0],
                                                         test_image_bands[index])
            log.info("band file: %s", test_band_file_name)
            test_image_band_sub_images[index].save(test_band_file_name, quality=90)
        test_image.save('%s/%s' % (output_head, input_tail), quality=90)
        # ann_output_data = list(ann_output.getdata())
        # flattened_ann_output_data = []
        # for pixel in ann_output_data:
        #     flattened_ann_output_data.append(flatten_and_rescale_pixel(pixel, debug=False))
        # log.info("flattened_ann_output_data %d pixels:", len(flattened_ann_output_data))
        # temp_string = ''
        # for pixel in flattened_ann_output_data:
        #     temp_string += '%7.6f ' % pixel
        # log.info('%s', temp_string)
