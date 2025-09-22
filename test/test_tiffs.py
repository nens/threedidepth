# -*- coding: utf-8 -*-

from pytest import mark

from osgeo.gdal import Open
from osgeo.osr import GetUserInputAsWKT

from threedidepth.tiffs import get_grid, AuxGeoTIFF


@mark.parametrize("bbox, width, height, geo_transform", [
    ((0.0, 0.0, 2.0, 2.0), 2, 2, (0.0, 1.0, 0, 2.0, 0, -1.0)),
    ((0.4, 0.0, 2.0, 2.0), 2, 2, (0.0, 1.0, 0, 2.0, 0, -1.0)),
    ((0.6, 0.0, 2.0, 2.0), 1, 2, (1.0, 1.0, 0, 2.0, 0, -1.0)),
    ((0.0, 0.4, 2.0, 2.0), 2, 2, (0.0, 1.0, 0, 2.0, 0, -1.0)),
    ((0.0, 0.6, 2.0, 2.0), 2, 1, (0.0, 1.0, 0, 2.0, 0, -1.0)),
    ((0.0, 0.0, 2.4, 2.0), 2, 2, (0.0, 1.0, 0, 2.0, 0, -1.0)),
    ((0.0, 0.0, 2.6, 2.0), 3, 2, (0.0, 1.0, 0, 2.0, 0, -1.0)),
    ((0.0, 0.0, 2.0, 2.4), 2, 2, (0.0, 1.0, 0, 2.0, 0, -1.0)),
    ((0.0, 0.0, 2.0, 2.6), 2, 3, (0.0, 1.0, 0, 3.0, 0, -1.0)),
])
def test_get_grid(bbox, width, height, geo_transform):
    grid = get_grid(bbox=bbox, origin=(9.0, 9.0), cellsize=(1.0, 1.0))
    assert grid.width == width
    assert grid.height == height
    assert grid.geo_transform == geo_transform


def test_get_grid2():
    bbox = 10, 40, 30, 60
    origin = 0.004, 0.004
    cellsize = 0.01, 0.01
    grid = get_grid(bbox=bbox, origin=origin, cellsize=cellsize)
    assert grid.width == 2000
    assert grid.height == 2000
    assert grid.geo_transform == (10.004, 0.01, 0.0, 60.004, 0.0, -0.01)


def test_aux_geo_tiff():
    bbox = 10, 40, 30, 60
    origin = 0.5, 0.5
    cellsize = 0.01, 0.01
    projection = GetUserInputAsWKT("EPSG:28992")

    with AuxGeoTIFF(
        bbox=bbox, origin=origin, cellsize=cellsize, projection=projection,
    ) as path:
        dataset = Open(path)
        assert (dataset.ReadAsArray() == -9999.0).all()
        assert dataset.GetProjection() == projection
        assert dataset.GetGeoTransform() == (
            bbox[0], cellsize[0], 0, bbox[3], 0, -cellsize[1],
        )
