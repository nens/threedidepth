# -*- coding: utf-8 -*-

from functools import reduce
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
