#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os.path


def add_filename_suffix(filename, suffix):
    """
    This function adds a suffix to the filename, keeping the extension.
    For example, if filename is test.nii.gz and suffix is "new",
    the returned name will be test_new.nii.gz
    :param filename: The full filename, including extension
    :param suffix: The suffix to add to the filename
    :return: The completed file name.
    """
    base, ext = split_name_with_nii(filename)

    return base + suffix + ext


def split_name_with_nii(filename):
    """
    Returns the clean basename and extension of a file.
    Means that this correctly manages the ".nii.gz" extensions.
    :param filename: The filename to clean
    :return: A tuple of the clean basename and the full extension
    """
    base, ext = os.path.splitext(filename)

    if ext == ".gz":
        # Test if we have a .nii additional extension
        temp_base, add_ext = os.path.splitext(base)

        if add_ext == ".nii":
            ext = add_ext + ext
            base = temp_base

    return (base, ext)


def listdir_fullpath(dir):
    """
    Lists the content of a directory with their full path.
    """

    return glob.glob(os.path.join(dir, "*"))


def case_insensitive_directory(parent, directory):
    """
    Finds if a directory exists, case insensitive.
    Returns the valid path, or False if it doesn't exist.
    """

    path = os.path.join(parent, directory)

    if os.path.isdir(path):
        return path
    else:
        # Try case insensitive search of this directory.
        try:
            parent_content = os.listdir(parent)
            lower_parent_content = [x.lower() for x in parent_content]
            lower_dir = directory.lower()

            dir_index = lower_parent_content.index(lower_dir)
            path = os.path.join(parent, parent_content[dir_index])

            if os.path.isdir(path):
                return path
            else:
                return False
        except (ValueError, OSError):
            return False


def list_files(dir):
    """
    Lists the files of a parent directory.
    """

    files = []

    for f in sorted(os.listdir(dir)):
        if os.path.isfile(os.path.join(dir, f)):
            files.append(f)

    return files


def list_subdirs(dir):
    """
    Lists the sub-directories of a parent directory.
    """

    subdirs = []

    for subdir in sorted(os.listdir(dir)):
        if os.path.isdir(os.path.join(dir, subdir)):
            subdirs.append(subdir)

    return subdirs
