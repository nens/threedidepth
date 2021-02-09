# -*- coding: utf-8 -*-

import pytest

from threedidepth import morton


def test_morton_array():
    result = morton.morton_array((8, 8)).tolist()
    expected = [[ 0,  1,  4,  5, 16, 17, 20, 21],  # noqa: E201
                [ 2,  3,  6,  7, 18, 19, 22, 23],  # noqa: E201
                [ 8,  9, 12, 13, 24, 25, 28, 29],  # noqa: E201
                [10, 11, 14, 15, 26, 27, 30, 31],
                [32, 33, 36, 37, 48, 49, 52, 53],
                [34, 35, 38, 39, 50, 51, 54, 55],
                [40, 41, 44, 45, 56, 57, 60, 61],
                [42, 43, 46, 47, 58, 59, 62, 63]]
    assert result == expected
    assert morton.morton_array((2,)).tolist() == [0, 1]
    assert morton.morton_array((1, 2)).tolist() == [[0, 1]]
    with pytest.raises(ValueError):
        morton.morton_array((2 ** 32, 2 ** 32))
