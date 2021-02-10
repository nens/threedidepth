# -*- coding: utf-8 -*-

import pytest

import numpy as np

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


def test_group():
    array = np.array([0.6, 0.6, 0.6, 0.5, 0.7, 0.5, 0.7, 0.5, 0.7])
    expected = [[3, 5, 7], [0, 1, 2], [4, 6, 8]]
    result = [i.tolist() for i in morton.group(array)]
    assert result == expected


def test_rasterize():
    # single point
    points = np.array([[0, 0]])
    expected = [[0]]
    result = morton.rasterize(points)[0].tolist()
    assert result == expected

    # horizontal
    points = np.array([[0, 0], [1, 0]])
    expected = [[0, 1]]
    result = morton.rasterize(points)[0].tolist()
    assert result == expected

    # vertical
    points = np.array([[0, 0], [1, 0]])
    expected = [[0, 1]]
    result = morton.rasterize(points)[0].tolist()
    assert result == expected

    # quadtree
    points = np.array([[1, 1], [3, 1], [1, 3], [3, 3], [6, 2]])
    expected = [
        [2, 3, 4],
        [0, 1, 5],
    ]
    result = morton.rasterize(points)[0].tolist()
    assert result == expected


def test_reorder():
    points = np.array([[1, 3], [3, 1], [6, 2], [3, 3], [1, 1]])
    s1 = np.array([1, 2, 3, 4, 5])

    result_points, result_s1 = morton.reorder(points, s1)

    expected_points = [[1, 3], [3, 3], [1, 1], [3, 1], [6, 2]]
    expected_s1 = [1, 4, 5, 2, 3]

    assert result_points.tolist() == expected_points
    assert result_s1.tolist() == expected_s1
