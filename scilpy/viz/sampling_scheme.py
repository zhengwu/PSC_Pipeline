from __future__ import division

import numpy as np

from dipy.data import get_sphere
from dipy.viz import fvtk

# TODO: Make it robust to more than 10 b-values
vtkcolors = [fvtk.colors.blue, 
             fvtk.colors.red,
             fvtk.colors.yellow, 
             fvtk.colors.purple, 
             fvtk.colors.cyan, 
             fvtk.colors.green,
             fvtk.colors.orange,
             fvtk.colors.white,
             fvtk.colors.brown,
             fvtk.colors.grey]

def plot_each_shell(ms, use_sym = True, use_sphere = True, same_color = False, rad = 0.025):
    if use_sphere:
        sphere = get_sphere('symmetric724')

    for i, shell in enumerate(ms):
        if same_color:
            i = 0
        ren = fvtk.ren()
        ren.SetBackground(1, 1, 1)
        if use_sphere:
            sphere_actor = fvtk.sphere_funcs(
                            np.ones(sphere.vertices.shape[0]), 
                            sphere, colormap='winter',  scale = 0)
            fvtk.add(ren, sphere_actor)
        pts_actor = fvtk.point(shell, vtkcolors[i], point_radius=rad)
        fvtk.add(ren, pts_actor)
        if use_sym:
            pts_actor = fvtk.point(-shell, vtkcolors[i], point_radius=rad)
            fvtk.add(ren, pts_actor)
        fvtk.show(ren)


def plot_proj_shell(ms, use_sym = True, use_sphere = True, same_color = False, rad = 0.025):

    ren = fvtk.ren()
    ren.SetBackground(1, 1, 1)
    if use_sphere:
        sphere = get_sphere('symmetric724')
        sphere_actor = fvtk.sphere_funcs(
                        np.ones(sphere.vertices.shape[0]), 
                        sphere, colormap='winter',  scale = 0)
        fvtk.add(ren, sphere_actor)

    for i, shell in enumerate(ms):
        if same_color:
            i = 0
        pts_actor = fvtk.point(shell, vtkcolors[i], point_radius=rad)
        fvtk.add(ren, pts_actor)
        if use_sym:
            pts_actor = fvtk.point(-shell, vtkcolors[i], point_radius=rad)
            fvtk.add(ren, pts_actor)
    fvtk.show(ren)


def build_shell_idx_from_bval(bvals, shell_th = 50):
    target_bvalues = _find_target_bvalues(bvals, shell_th = shell_th)

    # Pop b0
    if target_bvalues[0] < shell_th:
        target_bvalues.pop(0)

    shell_idx = _find_shells(bvals, target_bvalues, shell_th = shell_th)

    return shell_idx


def build_ms_from_shell_idx(bvecs, shell_idx):
    S = len(set(shell_idx))
    if (-1 in set(shell_idx)):
        S -= 1

    ms = []
    for i_ms in range(S):
        ms.append(bvecs[shell_idx==i_ms])

    return ms


# Attempt to find the b-values of the shells
def _find_target_bvalues(bvals, shell_th = 50):
    # Not robust
    target_bvalues = []

    bvalues = np.sort(np.array(list(set(bvals))))

    for bval in bvalues:
        add_bval = True
        for target_bval in target_bvalues:
            if (bval <= target_bval + shell_th) & (bval >= target_bval - shell_th):
                add_bval = False
        if add_bval:
            target_bvalues.append(bval)

    return target_bvalues


# Assign bvecs to a target shell
def _find_shells(bvals, target_bvalues, shell_th = 50):
    # Not robust
    # shell -1 means nbvecs not part of target_bvalues
    shells = -1 * np.ones_like(bvals)

    for shell_id, bval in enumerate(target_bvalues):
        shells[(bvals <= bval + shell_th) & (bvals >= bval - shell_th)] = shell_id

    return shells
