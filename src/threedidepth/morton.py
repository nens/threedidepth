# -*- coding: utf-8 -*-

from functools import reduce
from itertools import zip_longest
from math import ceil
from math import floor
from math import log

from scipy import ndimage
import numpy as np


def morton_array(shape):
    """
    Return array with Morton numbers.

    Inspired by:
    https://graphics.stanford.edu/%7Eseander/bithacks.html#InterleaveBMN
    """
    # determine the number of dimensions
    ndims = len(shape)

    # 1d compatibility
    if ndims == 1:
        return np.arange(shape[0])

    def bitcount(number):
        """ Return amount of bits used for in number """
        return int(ceil(log(number + 1, 2)))

    # feasbility check
    for i, j in enumerate(shape):
        # bit number assessment
        count = bitcount(j)                 # in the number
        count += (ndims - 1) * (count - 1)  # after spacing
        count += (ndims - 1) - i            # after shifting
        # numpy does not go higher than 64 bits currently
        if count > 64:
            raise ValueError('Too many bits needed for the computation')

    # generate list of zeros and masks
    ones = 1
    masks = []
    shifts = []
    pos = range(63, -1, -1)
    bmax = max(map(bitcount, shape))
    while ones < bmax:
        zeros = (ndims - 1) * ones
        shifts.append(zeros)
        period = ones + zeros
        masks.append(
            int(''.join('1' if i % period < ones else '0' for i in pos), 2),
        )
        ones *= 2

    # make indices and space them
    indices = [np.uint64(k) for k in np.ogrid[tuple(map(slice, shape))]]
    for i, (j, k) in enumerate(zip(shape, indices)):
        if j < 2:
            continue
        if j > 2:
            start = int(floor(log(bitcount(j) - 1, 2)))
        else:
            start = 0
        for n in range(start, -1, -1):
            k[:] = (k | k << shifts[n]) & masks[n]
        k <<= (ndims - 1) - i
    return reduce(np.bitwise_or, indices)


def get_morton_lut(array, no_data_value):
    """
    Return lookup table to rearrange an array of ints in morton order.

    :param array: 2D int array with a range of integers from 0 to no_data_value
    :param no_data_value: no data value that is excluded from rearrangement.

    The no_data_value does not have to be present in the array, but if it is,
    it does not get reordered by the lookup table (lut):
    lut[no_data_value] == no_data_value
    """
    # morton variables have underscores
    _array = morton_array(array.shape)
    _no_data_value = _array.max().item() + 1

    # make lookup from node to morton number
    index = np.arange(no_data_value + 1)
    lut1 = ndimage.minimum(_array, labels=array, index=index)
    lut1[no_data_value] = _no_data_value

    # make lookup from morton number back to node numbers
    lut2 = np.empty(_no_data_value + 1, dtype='i8')
    lut2[np.sort(lut1)] = index
    lut2[_no_data_value] = no_data_value

    # return the combined lookup table
    return lut2[lut1]


def group(array):
    """
    Return generator of arrays of indices to equal values.
    """
    order = array.argsort()
    _, index = np.unique(array[order], return_index=True)
    for start, stop in zip_longest(index, index[1:]):
        yield order[start:stop]


def analyze(x, y):
    """ Return (x_step, y_step) tuple.

    Return the smallest separation between points in the x-direction for points
    with the same y-coordinates and vice versa. That reveals the highest
    refinement level of the quadtree structure.
    """
    assert x.dtype == float
    assert y.dtype == float

    init = {'initial': np.inf}

    xs = min(np.diff(np.sort(x[i])).min(**init) for i in group(y))
    ys = min(np.diff(np.sort(y[i])).min(**init) for i in group(x))

    return None if np.isinf(xs) else xs, None if np.isinf(ys) else ys


def rasterize(points):
    """ Return (array, no_data_value) tuple.

    Rasterize the indices of the points in an array at the highest quadtree
    resolution. Note that points of larger squares in the quadtree also just
    occupy one cell in the resulting array, the rest of the cells get the
    no_data_value.
    """
    points = np.asarray(points, dtype=float)
    x, y = points.transpose()
    xs, ys = analyze(x, y)
    x1, y2 = x.min(), y.max()

    # get indices to land each point index in its own array cell
    j = np.int64(np.zeros_like(x) if xs is None else (x - x1) / xs)
    i = np.int64(np.zeros_like(y) if ys is None else (y2 - y) / ys)

    index = i, j
    no_data_value = len(points)
    ids = np.arange(no_data_value)

    values = np.full((i.max() + 1, j.max() + 1), no_data_value)
    values[index] = ids

    return values, no_data_value


def reorder(points, s1):
    """
    Return (points, s1) reordered to morton order.
    """
    array, no_data_value = rasterize(points)

    # array[lut] would have the ids in array in morton order
    lut = get_morton_lut(array=array, no_data_value=no_data_value)

    # the points need to be reordered such that rasterize(points[inv]) becomes
    # equal to lut[rasterize(points)] - in other words, for 'index value' a in
    # the raster to become 'index value' b in the raster, the index b in the
    # reordered points array must be occupied by the point from index a in the
    # old points array
    inv = np.arange(no_data_value)
    inv[lut[inv]] = inv.copy()  # may get bogus results without the copy

    return points[inv], s1[inv]
