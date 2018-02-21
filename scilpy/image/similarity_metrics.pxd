from __future__ import division
cimport cython
cimport libc.math as math
cimport numpy as cnp


# Computes a sum of values within 2d data
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline double sum_2d(double [:, ::1] data) nogil:
    cdef:
        double sum = 0.0
    
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            sum += data[i, j]
    
    return sum


# Computes a sum of values within 3d data
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline double sum_3d(double [:, :, ::1] data) nogil:
    cdef:
        double sum = 0.0
    
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            for k in range(0, data.shape[2]):
                sum += data[i, j, k]
    
    return sum


# Computes the mean value within 2d data
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline double mean_2d(double [:, ::1] data) nogil:
    return sum_2d(data) / (data.shape[0]*data.shape[1])


# Computes the mean value within 3d data
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline double mean_3d(double [:, :, ::1] data) nogil:
    return sum_3d(data) / (data.shape[0]*data.shape[1]*data.shape[2])


# Computes the local cross correlation of a 2d image at a given coordinate
# within a given square neighborhood
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline double xcorrelation_2d(double [:, ::1] data1,
                                 double [:, ::1] data2,
                                 cnp.npy_intp i,
                                 cnp.npy_intp j,
                                 cnp.npy_intp r) nogil:
    cdef:
        double [:, ::1] square1 = data1[i - r + 1 : i + r,
                                          j - r + 1 : j + r]
        double [:, ::1] square2 = data2[i - r + 1 : i + r,
                                          j - r + 1 : j + r]
        double a = 0.0, b = 0.0, c = 0.0, d, e, mean1, mean2
        
    mean1 = mean_2d(square1)
    mean2 = mean_2d(square2)
      
    for i in range(0, square1.shape[0]):
        for j in range(0, square1.shape[1]):
            a += (square1[i, j] - mean1) * (square2[i, j] - mean2)
            b += (square1[i, j] - mean1) * (square1[i, j] - mean1)
            c += (square2[i, j] - mean2) * (square2[i, j] - mean2)
                
    if b == 0.0 or c == 0.0:
        e = 0.0
    else:
        d = math.sqrt(b * c)
        e = math.fabs(a / d)
    
    return e


# Computes the local cross correlation of a 3d volume at a given coordinate
# within a given cube neighborhood
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline double xcorrelation_3d(double [:, :, ::1] data1,
                            double [:, :, ::1] data2,
                            cnp.npy_intp i,
                            cnp.npy_intp j,
                            cnp.npy_intp k,
                            cnp.npy_intp r) nogil:
    cdef:
        double [:, :, ::1] cube1 = data1[i - r + 1 : i + r,
                                          j - r + 1 : j + r,
                                          k - r + 1 : k + r]
        double [:, :, ::1] cube2 = data2[i - r + 1 : i + r,
                                          j - r + 1 : j + r,
                                          k - r + 1 : k + r]
        double a = 0.0, b = 0.0, c = 0.0, d, e, mean1, mean2
        
    mean1 = mean_3d(cube1)
    mean2 = mean_3d(cube2)
      
    for i in range(0, cube1.shape[0]):
        for j in range(0, cube1.shape[1]):
            for k in range(0, cube1.shape[2]):
                a += (cube1[i, j, k] - mean1) * (cube2[i, j, k] - mean2)
                b += (cube1[i, j, k] - mean1) * (cube1[i, j, k] - mean1)
                c += (cube2[i, j, k] - mean2) * (cube2[i, j, k] - mean2)
                
    if b == 0.0 or c == 0.0:
        e = 0.0
    else:
        d = math.sqrt(b * c)
        e = math.fabs(a / d)
    
    return e
