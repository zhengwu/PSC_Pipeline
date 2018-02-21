#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scilpy.utils.nibabel_tools import get_data


def get_reference_info(reference):
    _, nib_file = get_data(reference, return_object=True)
    reference_shape = nib_file.get_shape()
    reference_affine = nib_file.affine

    return reference_shape, reference_affine


def assert_same_resolution(*images):
    if len(images) == 0:
        raise Exception("Can't check if images are of the same "
                        "resolution/affine. No image has been given")

    ref = get_reference_info(images[0])
    for i in images[1:]:
        shape, aff = get_reference_info(i)
        if not (ref[0] == shape) and (ref[1] == aff).any():
            raise Exception("Images are not of the same resolution/affine")
