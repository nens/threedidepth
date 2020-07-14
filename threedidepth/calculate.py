# -*- coding: utf-8 -*-

from itertools import product
from os import path

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from osgeo import gdal
from threedigrid.admin.gridresultadmin import GridH5ResultAdmin
from threedigrid.admin.constants import SUBSET_2D_OPEN_WATER
from threedigrid.admin.constants import NO_DATA_VALUE
from threedidepth.fixes import fix_gridadmin


GRIDADMIN_PATH = "var/testdata/12997/gridadmin.h5"
DEM_PATH = "var/testdata/12997/dem.tif"
RESULTS_3DI_PATH = "var/testdata/12997/results_3di.nc"
WATERDEPTH_PATH = "waterdepth.tif"


class Calculator:
    pass


class BaseCalculator(Calculator):
    def __init__(self, gridadmin_path, results_3di_path, calculation_step):
        self.gridadmin_path = gridadmin_path
        self.results_3di_path = results_3di_path
        self.calculation_step = calculation_step

    def __call__(self, offset, values, no_data_value):
        """Return result values array.

        Args:
            offset (int, int): (row, col)-tuple determining array position
            values (array): source values for the calculation
            no_data_value (scalar): source and result no_data_value.

        Override this method to implement a different calculation. The default
        implementation is to just return the values, effectively copying the
        source.

        Note that the no_data_value for the result has to correspond to the
        no_data_value argument.
        """

        return values

    def __enter__(self):
        self.gr = GridH5ResultAdmin(self.gridadmin_path, self.results_3di_path)
        self.cache = {}
        return self

    def __exit__(self, *args):
        self.gr = None
        self.cache = None


class ConstantLevelDepthCalculator(BaseCalculator):
    """Depth calculator using constant waterlevel in a grid cell.

    Args:
        dem_pixelsize (float): Size of dem pixel in projected coordinates
        dem_shape (int, int): Shape of the dem array.
        gridadmin_path (str): Path to gridadmin.h5 file.
        results_3di_path (str): Path to results_3di.nc file.
        calculation_step (int): Calculation step for the waterdepth.
    """
    PIXEL_MAP = "pixel_map"
    LOOKUP_S1 = "lookup_s1"

    def __init__(self, dem_pixelsize, dem_shape, **kwargs):
        super().__init__(**kwargs)
        self.dem_pixelsize = dem_pixelsize
        self.dem_shape = dem_shape

    def __call__(self, offset, values, no_data_value):
        """Return waterdepth array.

        """
        # get or create the pixel map
        if self.PIXEL_MAP in self.cache:
            pixel_map = self.cache[self.PIXEL_MAP]
        else:
            pixel_map = self.gr.grid.get_pixel_map(
                dem_pixelsize=self.dem_pixelsize, dem_shape=self.dem_shape,
            )
            self.cache[self.PIXEL_MAP] = pixel_map

        # get or create the s1 lookup (the waterlevel table)
        if self.LOOKUP_S1 in self.cache:
            lookup_s1 = self.cache[self.LOOKUP_S1]
        else:
            nodes = self.gr.nodes.subset(SUBSET_2D_OPEN_WATER)
            timeseries = nodes.timeseries(indexes=[self.calculation_step])
            data = timeseries.only("s1", "id").data
            lookup_s1 = np.full((data["id"]).max() + 1, NO_DATA_VALUE)
            lookup_s1[data["id"]] = data["s1"]
            self.cache[self.LOOKUP_S1] = lookup_s1

        # determine waterlevel
        i1, j1 = offset
        h, w = values.shape
        i2, j2 = i1 + h, j1 + w
        waterlevel = lookup_s1[pixel_map[i1:i2, j1:j2]]

        # determine depth
        dem = values
        depth = np.full_like(dem, no_data_value)
        dem_active = (dem != no_data_value)
        waterlevel_active = (waterlevel != NO_DATA_VALUE)
        active = dem_active & waterlevel_active
        depth_1d = waterlevel[active] - values[active]

        # paste positive depths
        negative_1d = (depth_1d <= 0)
        depth_1d[negative_1d] = no_data_value
        depth[active] = depth_1d

        return depth

        # new procedure, gives error
        bbox_pix = [offset] + [o + s for (o, s) in zip(offset, values.shape)]
        nodgrid = self.gr.cells.get_nodgrid(
            bbox_pix, subset_name=SUBSET_2D_OPEN_WATER,
        )


class InterpolatedLevelDepthCalculator(BaseCalculator):
    """Depth calculator that interpolates waterlevel linearly between nodes.

    Args:
        dem_geo_transform: (tuple) Geo_transform of the dem.
        gridadmin_path (str): Path to gridadmin.h5 file.
        results_3di_path (str): Path to results_3di.nc file.
        calculation_step (int): Calculation step for the waterdepth.
    """
    INTERPOLATOR = "interpolator"

    def __init__(self, dem_geo_transform, **kwargs):
        super().__init__(**kwargs)
        self.dem_geo_transform = dem_geo_transform

    def __call__(self, offset, values, no_data_value):
        """Do:
        get node centers and s1, make an NDInterpolator object
        no_data? Only defined nodes?
        Evaluate at this grid (need coordinates for this piece)
        """
        # get or create the interpolator
        if self.INTERPOLATOR in self.cache:
            interpolator = self.cache[INTERPOLATOR]
        else:
            nodes = self.gr.nodes.subset(SUBSET_2D_OPEN_WATER)
            timeseries = nodes.timeseries(indexes=[self.calculation_step])
            data = timeseries.only("s1", "coordinates").data
            points = None
            values = None
            import ipdb
            ipdb.set_trace()
            interpolator = LinearNDInterpolator(
                points,
                values,
                fill_value=no_data_value
            )

        return values


class TiffConverter:
    """Convert tiff, applying a calculating function to the data.

    """
    def __init__(self, source_path, target_path):
        self.source_path = source_path
        self.target_path = target_path

    def __enter__(self):
        if path.exists(self.target_path):
            raise OSError("%s already exists." % self.target_path)
        self.source = gdal.Open(self.source_path, gdal.GA_ReadOnly)

        self.target = gdal.GetDriverByName("gtiff").Create(
            self.target_path,
            self.raster_x_size,
            self.raster_y_size,
            1,  # band count
            self.source.GetRasterBand(1).DataType,
            options=[
                "compress=deflate",
                "blockxsize=%s" % self.block_size[0],
                "blockysize=%s" % self.block_size[1],
            ],
        )
        self.target.SetProjection(self.projection)
        self.target.SetGeoTransform(self.geo_transform)
        self.target.GetRasterBand(1).SetNoDataValue(self.no_data_value)

        return self

    def __exit__(self, *args):
        """Close datasets.

        TODO check if actually needed
        """
        self.source = None
        self.target = None

    @property
    def projection(self):
        return self.source.GetProjection()

    @property
    def geo_transform(self):
        return self.source.GetGeoTransform()

    @property
    def no_data_value(self):
        return self.source.GetRasterBand(1).GetNoDataValue()

    @property
    def raster_x_size(self):
        return self.source.RasterXSize

    @property
    def raster_y_size(self):
        return self.source.RasterYSize

    @property
    def block_size(self):
        return self.source.GetRasterBand(1).GetBlockSize()

    def __len__(self):
        block_size = self.block_size
        blocks_x = -(-self.raster_x_size // block_size[0])
        blocks_y = -(-self.raster_y_size // block_size[1])
        return blocks_x * blocks_y

    def partition(self):
        """Return generator of (xoff, xsize), (yoff, ysize) values.
        """
        def offset_size_range(stop, step):
            for start in range(0, stop, step):
                yield start, min(step, stop - start)

        raster_size = self.raster_x_size, self.raster_y_size
        block_size = self.block_size
        generator = product(*map(offset_size_range, raster_size, block_size))

        total = len(self)
        for count, result in enumerate(generator, start=1):
            yield result
            gdal.TermProgress_nocb(count / total)

    def convert_using(self, calculator):
        """Convert data writing it to tiff. """
        if not isinstance(calculator, Calculator):
            raise TypeError("calculator must be of the Calculator type.")
        no_data_value = self.no_data_value
        for (xoff, xsize), (yoff, ysize) in self.partition():
            values = self.source.ReadAsArray(
                xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize,
            )

            result = calculator(
                offset=(yoff, xoff),
                values=values,
                no_data_value=no_data_value,
            )

            self.target.GetRasterBand(1).WriteArray(
                array=result, xoff=xoff, yoff=yoff
            )


def calculate_waterdepth(
    gridadmin_path=GRIDADMIN_PATH,
    results_3di_path=RESULTS_3DI_PATH,
    dem_path=DEM_PATH,
    waterdepth_path=WATERDEPTH_PATH,
    calculation_step=-1,
    interpolate=False,
):
    """Calculate waterdepth and save it as GeoTIFF.

    Args:
        gridadmin_path (str): Path to gridadmin.h5 file.
        results_3di_path (str): Path to results_3di.nc file.
        dem_path (str): Path to dem.tif file.
        waterdepth_path (str): Path to waterdepth.tif file.
        calculation_step (int): Calculation step for the waterdepth.
        interpolate (bool): Interpolate linearly between nodes.
    """
    fix_gridadmin(GRIDADMIN_PATH)  # TODO newer versions of the h5 are fixed.

    converter_kwargs = {
        "source_path": dem_path,
        "target_path": waterdepth_path,
    }

    # CalculatorClass = BaseCalculator

    with TiffConverter(**converter_kwargs) as converter:
        calculator_kwargs = {
            "gridadmin_path": gridadmin_path,
            "results_3di_path": results_3di_path,
            "calculation_step": calculation_step,
        }
        if interpolate:
            CalculatorClass = InterpolatedLevelDepthCalculator
            calculator_kwargs['dem_geo_transform'] = converter.geo_transform
        else:
            CalculatorClass = ConstantLevelDepthCalculator
            calculator_kwargs['dem_pixelsize'] = converter.geo_transform[1]
            calculator_kwargs['dem_shape'] = (
                converter.raster_y_size, converter.raster_x_size,
            )
        with CalculatorClass(**calculator_kwargs) as calculator:
            converter.convert_using(calculator)


calculate_waterdepth(interpolate=True)
