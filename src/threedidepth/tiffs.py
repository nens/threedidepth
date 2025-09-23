from contextlib import contextmanager
from threading import current_thread
from collections import namedtuple

from osgeo.gdal import (
    GDT_Float32,
    GetDriverByName,
    Unlink,
    VSIStatL,
)

OPTIONS = [
    "COMPRESS=DEFLATE",
    "SPARSE_OK=YES",
    "BIGTIFF=YES",
    "TILED=YES",
]
DRIVER = GetDriverByName("GTiff")
NO_DATA_VALUE = -9999.


def get_grid(bbox, origin, cellsize):
    x1, y1, x2, y2 = bbox
    dx, dy = cellsize

    # round to unbounded grid defined by origin and cellsize
    X, Y = origin

    x1 = round((x1 - X) / dx) * dx + X
    y1 = round((y1 - Y) / dy) * dy + Y
    x2 = round((x2 - X) / dx) * dx + X
    y2 = round((y2 - Y) / dy) * dy + Y

    # determine height, with and geo_transform
    width = round((x2 - x1) / dx)
    height = round((y2 - y1) / dy)
    assert width > 0 and height > 0
    geo_transform = x1, dx, 0, y2, 0, -dy

    # return
    Grid = namedtuple('Grid', ["width", "height", "geo_transform"])
    return Grid(width=width, height=height, geo_transform=geo_transform)


@contextmanager
def AuxGeoTIFF(bbox, origin, cellsize, projection):
    """
    Return in-memory dummy tiff. It is filled with no data, but it has appropriate
    chunking and geospatial properties to be used as prototype for calculations.
    """
    path = f"/vsimem/{current_thread().native_id}"
    grid = get_grid(bbox=bbox, origin=origin, cellsize=cellsize)
    dataset = DRIVER.Create(
        path,
        grid.width,
        grid.height,
        bands=1,
        eType=GDT_Float32,
        options=OPTIONS,
    )

    dataset.SetProjection(projection)
    dataset.SetGeoTransform(grid.geo_transform)

    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(NO_DATA_VALUE)

    # manually dereference the dataset object,
    # since it is used in the calling context
    dataset = None

    yield path

    # cleanup
    assert VSIStatL(path) is not None
    Unlink(path)
    assert VSIStatL(path) is None
