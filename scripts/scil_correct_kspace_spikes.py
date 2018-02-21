#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import argparse
import logging
import os

from dipy.io.gradients import read_bvals_bvecs
import nibabel as nib
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.cluster.vq import ClusterError
from scipy.cluster.vq import kmeans2 as kmeans

# The minimum value of the product of two bvecs for them to be considered
# neighbors.
MINIMUM_INNER = 0.7


def build_args_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='This script detects and removes k-space spikes in '
                    'diffusion weighted images. It detects \nthe spikes by '
                    'looking for oulier intensities in k-space blocks and '
                    'removes them by \nreplacing them with averages of other '
                    'images.')

    parser.add_argument('input', action='store', metavar='DWI_FILE', type=str,
                        help='The file that contains the diffusion weighted '
                             'images to correct.')

    parser.add_argument('bvalues', action='store',
                        metavar='BVALS_FILE', type=str,
                        help='The file that contains the b-values in the FSL '
                             'format).')

    parser.add_argument('bvectors', action='store',
                        metavar='BVECS_FILE', type=str,
                        help='The file that contains the b-vectors in the FSL '
                             'format.')

    parser.add_argument('output', action='store',
                        metavar='OUTPUT_FILE', type=str,
                        help='The file where the corrected diffusion weighted '
                             'images are saved')

    parser.add_argument('-b', '--block-size', action='store',
                        metavar='size', type=int, default=10,
                        help='The block size to use when detecting '
                             'spikes. [%(default)s]')

    parser.add_argument('-n', '--nb-shells', action='store',
                        metavar='N', type=int, default=2,
                        help='The number of b-value shells of the data, '
                             'including the b0. [%(default)s]')

    parser.add_argument('-t', '--threshold', action='store', metavar='z-score',
                        type=float, default=3.0,
                        help='Threshold used to detect the z-score '
                             'outliers. [%(default)s]')

    parser.add_argument('-f', '--force', action='store_true', dest='force',
                        help='Force overwrite (overwrite output file if '
                             'present).')

    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                        help='Make the output verbose.')

    return parser


def correct_kspace_spike(img_data, bvalues, bvectors, threshold,
                      nb_shells, block_size):
    """Detect and correct k-space spikes

    Detects and corrects k-space spikes in diffusion weighted images. The spikes
    are detected by finding blocks of k-space which have an intensity well
    above other blocks. They are corrected by replacing the blocks with the
    average of its neighbors.

    Args:
        img_data (ndarray) : The array that contains the image to correct. Must
            be 4D with the last two dimensions being slices and gradient
            directions.
        bvalues (ndarray) : The b-values of the data.
        bvectors (ndarray) : The b-vectors of the data.
        threshold (float) : The z-score used to find outlier intensities.
        nb_shells (int) : The number of shells of the data.
        block_size (int) : The size of the blocks used to detect spikes.

    Returns:
        The image data with some k-space blocks modified to mask the detected
        k-space spikes.

    """

    # Assign a shell to each b-value.
    bvalues, shells = find_bvalue_shells(bvalues, nb_shells)

    # We work in kspace so compute the FFT along the encoding directions.
    kspace_data = fftshift(fft2(img_data, axes=(0, 1)), axes=(0, 1))

    # Process each shell independently.
    for shell_number in set(shells):
        on_shell = shells == shell_number

        if on_shell.sum() > 3:
            logging.info(
                'Processing b-value {}.'.format(bvalues[shell_number]))

            # Directions with close bvectors are considered neighbors.
            shell_bvectors = bvectors[on_shell, :]
            inner = np.abs(np.dot(shell_bvectors, shell_bvectors.T))
            neighbors = inner > MINIMUM_INNER

            # Process every slice and every block independently.
            shell = kspace_data[..., on_shell]
            for slice in slices(shell):
                for data_block in blocks(slice, block_size):
                    fix_block(data_block, neighbors, threshold)
            kspace_data[..., on_shell] = shell

        else:
            logging.info(
                'Skipping b-value {}, not enough directions.'
                .format(bvalues[shell_number]))

    # Back to image space.
    img_data = np.abs(ifft2(ifftshift(kspace_data, axes=(0, 1)), axes=(0, 1)))
    return img_data


def fix_block(data, neighbors, threshold):
    """Removes k-space spikes in a block of data

    Replaces blocks of data where k-space spikes are detected by the average of
    neighboring blocks.

    Args:
        data (ndarray) : The array that contains the blocks to process. The
            first two dimensions are the block size and the third is the number
            of blocks.
        neighbors (ndarray) : An array that contains True if two blocks are
            neighbors and False otherwise.
        threshold (float) : The threshold used to detect blocks with spikes.

    """

    is_fixed = False
    while not is_fixed:

        # The sum of each slice of data is used to detect the spike.
        data_sum = np.abs(data).sum(axis=(0, 1))
        z_scores = (data_sum - data_sum.mean()) / data_sum.std()

        # Any direction with a intensity above the threshold is considered
        # problematic.
        if np.max(z_scores) > threshold:

            direction = np.argmax(z_scores)

            # Replace the problem block with the mean of the neighbors that
            # have a good z-score.
            good_neighbors = np.logical_and(
                neighbors[direction],
                np.abs(z_scores) < 1)
            mean_block = data[:, :, good_neighbors].mean(axis=2)
            data[:, :, direction] = mean_block

        else:
            is_fixed = True


def slices(data):
    """Generator that iterates on slices of data"""
    for i in range(data.shape[2]):
        yield data[:, :, i, ...]


def blocks(data, size):
    """Generator used to traverse data in overlapping blocks"""
    for i in range(0, data.shape[0] - size - 1, size // 2):
        for j in range(0, data.shape[1] - size - 1, size // 2):
            yield data[i:i+size, j:j+size, ...]


def find_bvalue_shells(bvals, nb_shells):
    """Finds the b-value of each shell and the shell of each b-value"""

    # We basically want to classify the b-values so k-means will work.
    # Sometimes one group is empty so we try again a few times.
    nb_repeats = 0
    while True:

        try:
            bvalues, shells = kmeans(bvals, nb_shells, missing='raise')
            break

        except ClusterError:
            nb_repeats += 1
            if nb_repeats > 4:
                raise ValueError(
                    'Could not identity the b-value for each shell. Are you '
                    'sure the number of shells is {}?'.format(nb_shells))

    logging.info(
        'The shells have a b-value of {}.'
        .format(bvalues.astype(int)))
    return bvalues.astype(int), shells


def main():

    parser = build_args_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if os.path.isfile(args.output):
        if args.force:
            logging.info('Overwriting {0}.'.format(args.output))
        else:
            parser.error(
                '{0} already exist! Use -f to overwrite it.'
                .format(args.output))

    # Load the input data.
    img = nib.load(args.input)
    img_data = img.get_data()
    bvalues, bvectors = read_bvals_bvecs(args.bvalues, args.bvectors)

    # Detect and remove the spikes.
    new_img_data = correct_kspace_spike(
        img_data, bvalues, bvectors,
        args.threshold, args.nb_shells, args.block_size)

    # Save the corrected image.
    new_img = nib.Nifti1Image(
        new_img_data.astype(img_data.dtype),
        img.get_affine())
    nib.save(new_img, args.output)


if __name__ == "__main__":
    main()
