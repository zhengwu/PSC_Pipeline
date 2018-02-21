#! /usr/bin/env python
# -*- coding: utf-8 -*-

DESCRIPTION = """
    Show a DWI animation along a certain axis
    """

import argparse
import logging as log

from dipy.segment import mask
import imageio
from matplotlib import animation
from matplotlib import pyplot as plt
import nibabel as nb
import numpy as np


class DWI_Animater:
    _data = []  # Input data
    _axis = 0  # Selected axis
    _qVolume = []  # Q-Space volume slice
    _visuVolume = []  # Visualization volume
    _currentFrame = []  # Current animation frame
    _currentText = ''  # Current displayed text
    _currentPosition = [0, 0, 0, 0]  # Current position of the cursor
    _currentMean = 0  # Mean of all Q-Space values at current position
    _pause = False  # Animation pause status

    # Init function
    def __init__(self, filename, axis='axial', slicing=None):
        # Load data
        data = nb.load(filename)
        self._data = data.get_data()

        # Verify axis
        axes = {'sagittal': 0, 's': 0, 'coronal': 1, 'c': 1, 'axial': 2, 'a': 2}
        if not(axis in axes):
            log.critical("Wrong axis. Given " + axis +
                         " and takes only sagittal, coronal or axial.")
            raise ValueError()
        self._axis = axes[axis]

        # Verify slicing
        slice_size = self._data.shape[self._axis]
        if slicing is None:
            slicing = slice_size / 2
        elif slicing < 0 or slicing >= slice_size:
            log.critical("Slice outside volume limits. Given " + str(slicing) +
                         " and takes only in range " + str([0, slice_size-1]))
            raise ValueError()
        self._currentPosition[3] = int(slicing)

        # Select axis and slicing and rearrange axes for proper viewing
        if self._axis == 0:
            self._qVolume = np.swapaxes(
                np.swapaxes(self._data, 1, 2), 0, 1)[::-1, slicing, :, :]
        elif self._axis == 1:
            self._qVolume = np.swapaxes(
                self._data, 0, 2)[::-1, slicing, ::-1, :]
        elif self._axis == 2:
            self._qVolume = np.swapaxes(self._data, 1, 0)[::-1, :, slicing, :]

    # Mouse move callback function
    def _onMove(self, _event):
        if _event.xdata is not None and _event.ydata is not None:
            if _event.xdata < 0:
                self._currentPosition[1] = 0
            elif _event.xdata >= self._qVolume.shape[1]:
                self._currentPosition[1] = int(self._qVolume.shape[1]) - 1
            else:
                self._currentPosition[1] = int(_event.xdata)

            if _event.ydata < 0:
                self._currentPosition[0] = 0
            elif _event.ydata >= self._qVolume.shape[0]:
                self._currentPosition[0] = self._qVolume.shape[0] - 1
            else:
                self._currentPosition[0] = int(_event.ydata)

            self._currentMean = int(np.mean(
                self._qVolume[
                    self._currentPosition[0],
                    self._currentPosition[1],
                    :]))

    # Mouse click callback function
    def _onClick(self, _event):
        if self._pause:
            self._pause = False
        else:
            self._pause = True

    # Mouse scroll callback function
    def _onScroll(self, _event):
        if self._pause:
            if self._currentPosition[2] + int(_event.step) < 0:
                self._currentPosition[2] = 0
            elif self._currentPosition[2] + int(_event.step) >= int(self._qVolume.shape[2]):
                self._currentPosition[2] = int(self._qVolume.shape[2] - 1)
            else:
                self._currentPosition[2] += int(_event.step)

    # Q-Space animation function
    def _updateImage(self, _frame):
        if not self._pause:
            self._currentPosition[2] = int(_frame)
        self._currentFrame.set_array(
            self._visuVolume[:, :, self._currentPosition[2]])
        currentPosition = np.copy(self._currentPosition)
        currentPosition[self._axis] = self._currentPosition[3]
        currentPosition[3] = self._currentPosition[self._axis]
        self._currentText.set_text(
            "Position: Value, Mean\n" +
            str(currentPosition) +
            " : " +
            str(self._qVolume[self._currentPosition[0],
                self._currentPosition[1],
                self._currentPosition[2]]) +
            ", " +
            str(self._currentMean))
        return self._currentFrame

    def show(self, rescale=False, fps=10, output=''):
        self._visuVolume = self._qVolume.astype("uint8")

        # Scale data direction wise or globally
        if rescale:
            log.info("Scaling data for every direction individually...")
            for i in range(0, self._qVolume.shape[2]):
                diff_dir_image = np.copy(self._qVolume[:, :, i].astype(float))
                mini, maxi = np.amin(diff_dir_image), np.amax(diff_dir_image)
                self._visuVolume[:, :, i] = (255.0*(
                    diff_dir_image-mini)/maxi).astype("uint8")
        else:
            image = np.copy(self._qVolume.astype(float))
            mini, maxi = np.amin(image), np.amax(image)
            self._visuVolume = (255.0*(image-mini)/maxi).astype("uint8")

        if output:
            # Output the animation
            images = self._visuVolume
            crop_image = np.copy(images)

            # Cropping volume
            t = 0.05
            log.info("Cropping volume in a bounding box from pixels below " +
                     str(int(t*100.0)) + " % intensity threshold ...")
            crop_image[crop_image <= int(t*255.0)] = 0
            min_bounds, max_bounds = mask.bounding_box(crop_image)
            images = mask.crop(images, min_bounds, max_bounds)

            # Swap axes for gif writer
            images = np.swapaxes(images, 0, 2)
            images = np.swapaxes(images, 1, 2)

            if output.find(".gif") < 0:
                output = output + ".gif"
            imageio.mimwrite(output, images)
            log.info(output + " file created.")
        else:
            # Initialize figure
            fig = plt.figure()

            # Add current frame to figure
            plt.subplot(121)
            self._currentFrame = plt.imshow(
                self._visuVolume[:, :, 0],
                cmap='gray')
            if not rescale:
                plt.colorbar()
            self._currentFrame.set_interpolation('nearest')
            plt.axis('off')

            # Add text display to figure
            plt.subplot(122)
            plt.text(
                0.5,
                0.7,
                "Data : " +
                str(self._data.shape) +
                "\n" +
                "Q-Space volume : " +
                str(self._qVolume.shape),
                horizontalalignment='center',
                verticalalignment='center')
            self._currentText = plt.text(
                0.5,
                0.5,
                "Position , action='store_true': Value, Mean\n" +
                str(self._currentPosition),
                horizontalalignment='center',
                verticalalignment='center',
                fontsize='x-large')
            plt.axis('off')

            # Connect the callback functions
            fig.canvas.mpl_connect('motion_notify_event', self._onMove)
            fig.canvas.mpl_connect('button_press_event', self._onClick)
            fig.canvas.mpl_connect('scroll_event', self._onScroll)

            # Create the animation in the figure
            anim = animation.FuncAnimation(
                fig,
                self._updateImage,
                frames=self._qVolume.shape[2],
                interval=1000.0 / fps)

            plt.show()


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('input', action='store', type=str,
                   help='Path of the diffusion volume')
    p.add_argument('-a', metavar="axis", dest="axis", action='store', type=str,
                   choices=['s', 'sagittal', 'c', 'coronal', 'a', 'axial'],
                   default='axial', help='Axis to analyze, sagittal, coronal or \
                   axial [axial]')
    p.add_argument('-s', metavar="slice", dest="slice", action='store', type=int,
                   default=None, help='Volume slice to analyze [none] (middle slice)')
    p.add_argument('-r', metavar="rate", dest="frame_rate", action='store', type=int,
                   default=10, help='Animation frame rate per second [10]')
    p.add_argument('-o', metavar="filename", dest="output_filename", action='store',
                   type=str, default='', help='Output gif animation filename \
                   [none] (interactive mode)')
    p.add_argument('--rescale', action='store_true',
                   help='Rescale image for better viewing [none]')
    p.add_argument('-v', action='store_true',
                   help='Print program state messages.')
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    if args.v:
        log.getLogger().setLevel(log.INFO)
    dwi_animater = DWI_Animater(args.input, args.axis, args.slice)
    dwi_animater.show(args.rescale, args.frame_rate, args.output_filename)

if __name__ == '__main__':
    main()
