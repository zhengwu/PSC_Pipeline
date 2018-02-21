#! /usr/bin/env python

from __future__ import division, print_function

import argparse
import numpy as np

from dipy.io.gradients import read_bvals_bvecs
from dipy.viz import fvtk


DESCRIPTION = """
    Script to visualize bvecs schemes projected on a sphere.
    """


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('bvals', action='store', metavar='bvals',
                   help='Path of the bvals file, in FSL format.')

    p.add_argument('bvecs', action='store', metavar='bvecs',
                   help='Path of the bvecs file, in FSL format.')

    p.add_argument('--points', action='store_true', dest='points',
                   help='If set, show points instead of labels. (Default: False)')

    p.add_argument('--antipodal', action='store_true', dest='antipod',
                   help='If set, show antipodal points instead of labels. (Default: False)')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)

    ren = fvtk.ren()

    if args.points:
            points = fvtk.point(bvecs * bvals[..., None] * 0.01, 
                                fvtk.colors.red, point_radius=.5)
            fvtk.add(ren, points)

            if args.antipod :
                points = fvtk.point(-bvecs * bvals[..., None] * 0.01, 
                                     fvtk.colors.green, point_radius=.5)
                fvtk.add(ren, points)
    else:
        for i in range(bvecs.shape[0]):
            label = fvtk.label(ren, text=str(i), pos=bvecs[i]*bvals[i]*0.01, 
                               color=fvtk.colors.red, scale=(0.5, 0.5, 0.5))
            fvtk.add(ren, label)
            if args.antipod :
                label = fvtk.label(ren, text=str(i), pos=-bvecs[i]*bvals[i]*0.01, 
                                   color=fvtk.colors.green, scale=(0.5, 0.5, 0.5))
                fvtk.add(ren, label)

    fvtk.show(ren)


if __name__ == "__main__":
    main()
