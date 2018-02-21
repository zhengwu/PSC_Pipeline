#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import os

from dipy.core.sphere import Sphere
from dipy.utils.arrfuncs import as_native_array
from dipy.reconst.peaks import peak_directions
from dipy.reconst.shm import sph_harm_lookup, smooth_pinv


def find_order_from_nb_coeff(data):
    return int((-3 + np.sqrt(1 + 8 * data.shape[-1])) / 2)


def get_b_matrix(order, sphere, basis_type, return_all=False):
    sph_harm_basis = sph_harm_lookup.get(basis_type)
    if sph_harm_basis is None:
        raise ValueError("Invalid basis name.")
    b_matrix, m, n = sph_harm_basis(order, sphere.theta, sphere.phi)
    if return_all:
        return b_matrix, m, n
    return b_matrix


def get_maximas(data, sphere, b_matrix, threshold, absolute_threshold,
                min_separation_angle=25):
    spherical_func = np.dot(data, b_matrix.T)
    spherical_func[np.nonzero(spherical_func < absolute_threshold)] = 0.
    return peak_directions(
        spherical_func, sphere, threshold, min_separation_angle)


# TODO Delete when available on DiPy > 0.11
def get_repulsion200_sphere():
    file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../../data/repulsion200.npz')
    res = np.load(file_path)
    return Sphere(xyz=as_native_array(res['vertices']),
                  faces=as_native_array(res['faces']))


class SphericalHarmonics():

    def __init__(self, odf_data, basis, sphere):
        self.basis = basis
        self.order = find_order_from_nb_coeff(odf_data)
        self.sphere = sphere
        B, self.m, self.n = get_b_matrix(self.order, self.sphere, self.basis,
                                         return_all=True)
        self.B = np.ascontiguousarray(np.matrix(B), dtype=np.float64)
        invB = smooth_pinv(self.B, np.sqrt(0.006) * (-self.n * (self.n + 1)))
        self.invB = np.ascontiguousarray(np.matrix(invB), dtype=np.float64)

    def get_SF(self, sh):
        a = np.ascontiguousarray(sh, dtype=np.float64)
        SF = np.dot(self.B, a).reshape((np.shape(self.sphere.vertices)[0], 1))
        return SF
