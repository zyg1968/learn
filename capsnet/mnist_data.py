#! /usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import os

import numpy
import shutil
from six.moves import urllib

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
#SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'


def download_data(dir='data/mnist'):
    TRAIN_IMAGES = 'train-images-idx3-ubyte'
    TRAIN_LABELS = 'train-labels-idx1-ubyte'
    TEST_IMAGES = 't10k-images-idx3-ubyte'
    TEST_LABELS = 't10k-labels-idx1-ubyte'

    fn1 = maybe_download(TRAIN_IMAGES, dir, SOURCE_URL + TRAIN_IMAGES)
    #with open(local_file, 'rb') as f:
    #    train_images = extract_images(f)
    fn2 = maybe_download(TRAIN_LABELS, dir, SOURCE_URL + TRAIN_LABELS)
    fn3 = maybe_download(TEST_IMAGES, dir, SOURCE_URL + TEST_IMAGES)
    fn4 = maybe_download(TEST_LABELS, dir, SOURCE_URL + TEST_LABELS)
    return fn1,fn2,fn3,fn4

def maybe_download(filename, data_directory, source_url):
    """Download the data from source url, unless it's already here.

    Args:
        filename: string, name of the file in the directory.
        work_directory: string, path to working directory.
        source_url: url to download from if file doesn't exist.

    Returns:
        Path to resulting file.
    """
    if not os.path.exists(data_directory):
        print('Not found data directory, create directory ', data_directory)
        os.makedirs(data_directory)
    filepath = os.path.join(data_directory, filename)

    def download_progress(count, block_size, total_size):
        sys.stdout.write("\r>> Downloading %s %.1f%%" % (filename, float(count * block_size) / float(total_size) * 100.))
        sys.stdout.flush()

    if os.path.isfile(filepath):
        print("file {} already download and extracted.".format(filename))
        return filepath
    elif os.path.isfile(filepath+'.gz'):
        print("file {} already download, now extract it.".format(filepath+'.gz'))
    else:
        print('Not found {}, world downloaded from {}'.format(filepath, source_url))
        filepath, _ = six.moves.urllib.request.urlretrieve(source_url, filepath+'.gz', download_progress)
        print()
        print('Successfully Downloaded', filename)
    with gzip.open(filepath+'.gz', 'rb') as f_in, open(filepath, 'wb') as f_out:
        print('Extracting ', filename)
        shutil.copyfileobj(f_in, f_out)
        print('Successfully extracted')
    return filepath

