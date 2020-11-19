from itertools import product
from os import path
import numpy as np
import math
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
from osgeo import gdal
from threedigrid.admin.gridresultadmin import GridH5ResultAdmin
from threedigrid.admin.gridresultadmin import GridH5Admin
from threedigrid.admin.constants import SUBSET_2D_OPEN_WATER
from threedigrid.admin.constants import NO_DATA_VALUE
from threedidepth.fixes import fix_gridadmin
import time

MODE_COPY = "copy"
MODE_NODGRID = "nodgrid"
MODE_CONSTANT_S1 = "constant-s1"
MODE_INTERPOLATED_S1 = "interpolated-s1"
MODE_CONSTANT = "constant"
MODE_INTERPOLATED = "interpolated"
MODE_MY_INTERP_1 = "new interpolation triangles"
MODE_MY_INTERP_1_DEPTH = "new interpolation triangles depth"
MODE_MY_INTERP_2 = "new interpolation squares"
MODE_MY_INTERP_2_DEPTH = "new interpolation squares depth"

INTERPOLATION_METHOD = "distance"  # "distance"#"barycentric""#combi"
STEP = "0"


class Calculator:
    """Depth calculator using constant waterlevel in a grid cell.
    Args:
        gridadmin_path (str): Path to gridadmin.h5 file.
        results_3di_path (str): Path to results_3di.nc file.
        calculation_step (int): Calculation step for the waterdepth.
        dem_pixelsize (float): Size of dem pixel in projected coordinates
        dem_shape (int, int): Shape of the dem array.
        dem_geo_transform: (tuple) Geo_transform of the dem.
    """

    PIXEL_MAP = "pixel_map"
    LOOKUP_S1 = "lookup_s1"
    INTERPOLATOR = "interpolator"

    def __init__(
        self,
        gridadmin_path,
        results_3di_path,
        calculation_step,
        dem_shape,
        dem_geo_transform,
        dem_pixelsize,
    ):
        self.gridadmin_path = gridadmin_path
        self.results_3di_path = results_3di_path
        self.calculation_step = calculation_step
        self.dem_shape = dem_shape
        self.dem_geo_transform = dem_geo_transform
        self.dem_pixelsize = dem_pixelsize

    def __call__(self, indices, values, no_data_value):
        """Return result values array.
        Args:
            indices (tuple): ((i1, j1), (i2, j2)) subarray indices
            values (array): source values for the calculation
            no_data_value (scalar): source and result no_data_value
        Override this method to implement a calculation. The default
        implementation is to just return the values, effectively copying the
        source.
        Note that the no_data_value for the result has to correspond to the
        no_data_value argument.
        """
        raise NotImplementedError

    @staticmethod
    def _depth_from_water_level(dem, fillvalue, waterlevel):
        # determine depth
        depth = np.full_like(dem, fillvalue)
        dem_active = dem != fillvalue
        waterlevel_active = waterlevel != NO_DATA_VALUE
        active = dem_active & waterlevel_active
        depth_1d = waterlevel[active] - dem[active]

        # paste positive depths only
        negative_1d = depth_1d <= 0
        depth_1d[negative_1d] = fillvalue
        depth[active] = depth_1d

        return depth

    @property
    def lookup_s1(self):
        try:
            return self.cache[self.LOOKUP_S1]
        except KeyError:
            nodes = self.gr.nodes.subset(SUBSET_2D_OPEN_WATER)
            timeseries = nodes.timeseries(indexes=[self.calculation_step])
            data = timeseries.only("s1", "id").data
            lookup_s1 = np.full((data["id"]).max() + 1, NO_DATA_VALUE)
            lookup_s1[data["id"]] = data["s1"]
            self.cache[self.LOOKUP_S1] = lookup_s1
        return lookup_s1

    @property
    def interpolator(self):
        try:
            return self.cache[self.INTERPOLATOR]
        except KeyError:
            nodes = self.gr.nodes.subset(SUBSET_2D_OPEN_WATER)
            timeseries = nodes.timeseries(indexes=[self.calculation_step])
            data = timeseries.only("s1", "coordinates").data
            points = data["coordinates"].transpose()
            values = data["s1"][0]
            interpolator = LinearNDInterpolator(
                points, values, fill_value=NO_DATA_VALUE
            )
            self.cache[self.INTERPOLATOR] = interpolator
            return interpolator

    def _get_nodgrid(self, indices):
        """Return node grid.
        Args:
            indices (tuple): ((i1, j1), (i2, j2)) subarray indices
        """
        (i1, j1), (i2, j2) = indices

        # note that get_nodgrid() starts counting rows from the bottom
        h = self.dem_shape[0]
        i1, i2 = h - i2, h - i1

        # note that get_nodgrid() expects a columns-first bbox
        return self.gr.cells.get_nodgrid(
            [j1, i1, j2, i2], subset_name=SUBSET_2D_OPEN_WATER
        )

    def _get_points(self, indices):
        """Return points array.
        Args:
            indices (tuple): ((i1, j1), (i2, j2)) subarray indices
        """
        (i1, j1), (i2, j2) = indices
        local_ji = np.mgrid[i1:i2, j1:j2].reshape(2, -1)[::-1].transpose()

        p, a, b, q, c, d = self.dem_geo_transform
        return local_ji * [a, d] + [p + 0.5 * a, q + 0.5 * d]

    def __enter__(self):
        self.gr = GridH5ResultAdmin(self.gridadmin_path, self.results_3di_path)
        self.ga = GridH5Admin(self.gridadmin_path)
        self.cache = {}
        return self

    def __exit__(self, *args):
        self.gr = None
        self.cache = None


class MyCalculator:
    """New interpolation method for water depth.
    Args:
        gridadmin_path (str): Path to gridadmin.h5 file.
        results_3di_path (str): Path to results_3di.nc file.
        calculation_step (int): Calculation step for the waterdepth.
        dem_pixelsize (float): Size of dem pixel in projected coordinates
        dem_shape (int, int): Shape of the dem array.
        dem_geo_transform: (tuple) Geo_transform of the dem.
    """

    LOOKUP_NODES = "lookup_nodes"
    LOOKUP_LINES = "lookup_lines"
    NR = "nr_nodes_lines"

    def __init__(
        self,
        gridadmin_path,
        results_3di_path,
        calculation_step,
        dem_shape,
        dem_geo_transform,
        dem_pixelsize,
    ):
        self.gridadmin_path = gridadmin_path
        self.results_3di_path = results_3di_path
        self.calculation_step = calculation_step
        self.dem_shape = dem_shape
        self.dem_geo_transform = dem_geo_transform
        self.dem_pixelsize = dem_pixelsize

    def __call__(self, indices, values, no_data_value):
        """Return result values array.
        Args:
            indices (tuple): ((i1, j1), (i2, j2)) subarray indices
            values (array): source values for the calculation
            no_data_value (scalar): source and result no_data_value
        Override this method to implement a calculation. The default
        implementation is to just return the values, effectively copying the
        source.
        Note that the no_data_value for the result has to correspond to the
        no_data_value argument.
        """
        raise NotImplementedError

    @staticmethod
    def _depth_from_water_level(dem, fillvalue, waterlevel):
        # determine depth
        depth = np.full_like(dem, fillvalue)
        dem_active = dem != fillvalue
        waterlevel_active = waterlevel != NO_DATA_VALUE
        active = dem_active & waterlevel_active
        depth_1d = waterlevel[active] - dem[active]

        # paste positive depths only
        negative_1d = depth_1d <= 0
        depth_1d[negative_1d] = fillvalue
        depth[active] = depth_1d

        return depth

    def _get_nodgrid(self, indices):
        """Return node grid.
        Args:
            indices (tuple): ((i1, j1), (i2, j2)) subarray indices
        """
        (i1, j1), (i2, j2) = indices

        # note that get_nodgrid() starts counting rows from the bottom
        h = self.dem_shape[0]
        i1, i2 = h - i2, h - i1

        # note that get_nodgrid() expects a columns-first bbox
        return self.gr.cells.get_nodgrid(
            [j1, i1, j2, i2], subset_name=SUBSET_2D_OPEN_WATER
        )

    def _get_points(self, indices):
        """Return points array.
        Args:
            indices (tuple): ((i1, j1), (i2, j2)) subarray indices
        """
        (i1, j1), (i2, j2) = indices
        local_ji = np.mgrid[i1:i2, j1:j2].reshape(2, -1)[::-1].transpose()

        p, a, b, q, c, d = self.dem_geo_transform
        return local_ji * [a, d] + [p + 0.5 * a, q + 0.5 * d]

    def __enter__(self):
        self.gr = GridH5ResultAdmin(self.gridadmin_path, self.results_3di_path)
        self.ga = GridH5Admin(self.gridadmin_path)
        self.cache = {}
        return self

    def __exit__(self, *args):
        self.gr = None
        self.cache = None

    @property
    def lookup_nodes(self):
        try:
            return self.cache[self.LOOKUP_NODES]
        except KeyError:
            nodes = self.gr.nodes.subset(SUBSET_2D_OPEN_WATER)
            timeseries = nodes.timeseries(indexes=[self.calculation_step])
            data = timeseries.only("s1", "id", "coordinates", "cell_coords").data
            lookup_nodes = np.full(((data["id"]).max() + 1, 7), NO_DATA_VALUE)
            lookup_nodes[data["id"], 0] = data["s1"]
            lookup_nodes[data["id"], 1:3] = data["coordinates"].transpose()
            lookup_nodes[data["id"], 3:7] = data["cell_coords"].transpose()
            self.cache[self.LOOKUP_NODES] = lookup_nodes
        return lookup_nodes

    @property
    def lookup_lines(self):
        try:
            return self.cache[self.LOOKUP_LINES]
        except KeyError:
            lines = self.gr.lines.subset(SUBSET_2D_OPEN_WATER)
            timeseries_lines = lines.timeseries(indexes=[self.calculation_step])
            data_lines = timeseries_lines.only("au", "line_coords", "line").data
            lookup_lines = np.full(
                (len(data_lines["line"].transpose()), 7), NO_DATA_VALUE
            )
            lookup_lines[:, 0:2] = data_lines["line"].transpose()
            lookup_lines[:, 2:6] = data_lines["line_coords"].transpose()
            lookup_lines[:, 6:7] = data_lines["au"].transpose()
            self.cache[self.LOOKUP_LINES] = lookup_lines
        return lookup_lines

    @property
    def nr_nodes_lines(self):
        try:
            return self.cache[self.NR]
        except KeyError:
            nr_nodes = len(self.gr.nodes.subset(SUBSET_2D_OPEN_WATER).id)
            nr_lines = len(self.gr.lines.subset(SUBSET_2D_OPEN_WATER).au[-1, :])
            nr = [nr_nodes, nr_lines]
            self.cache[self.NR] = nr
        return nr

    def cell_info(self, indexX, coords, centres, lines_wrt_cells, AU, cell_id):
        """ 
        Return:
            cell_to_X_coords: coordinates of cell at position X wrt original cell.
            cell_to_X_centre: centre of cell at position X.
            au_cell_to_X: wet cross-sectional area of cell at position X.
        Args:
            indexX: index of cell at position X wrt original cell.
            coords (array): bottom-left and top-right coordinates of cells.
            centres (array): coordinates of cell-centres.
            lines_wrt_cells (array): cells that lines connects.
            AU (array): wet cross-sectional area at lines.
            cell_id: id of original cell.
        """
        indexX = np.array(indexX)
        if indexX.size == 0:  # no cell at position X
            cell_to_X_coords = [
                self.no_data_value,
                self.no_data_value,
                self.no_data_value,
                self.no_data_value,
            ]
            cell_to_X_centre = [self.no_data_value, self.no_data_value]
            au_cell_to_X = np.array(0)
        else:
            cell_to_X_coords = coords[:, indexX]
            cell_to_X_centre = [centres[indexX, 0], centres[indexX, 1]]
            cell_to_X_id = indexX + 1
            line_cell_to_X = np.array(indexX.size)
            au_cell_to_X = np.zeros(indexX.size)
            if indexX.size == 1:
                line_cell_to_X = np.where(
                    (
                        (lines_wrt_cells[0, :] == cell_to_X_id)
                        & (lines_wrt_cells[1, :] == cell_id)
                    )
                    | (
                        (lines_wrt_cells[0, :] == cell_id)
                        & (lines_wrt_cells[1, :] == cell_to_X_id)
                    )
                )
                if line_cell_to_X[0].size > 0:
                    au_cell_to_X = AU[line_cell_to_X]
                else:  # there is no line connecting original cell and cell at position X
                    au_cell_to_X = np.array([0])
            else:
                for j in range(indexX.size):
                    line_cell_to_X = np.where(
                        (
                            (lines_wrt_cells[0, :] == cell_to_X_id[j])
                            & (lines_wrt_cells[1, :] == cell_id)
                        )
                        | (
                            (lines_wrt_cells[0, :] == cell_id)
                            & (lines_wrt_cells[1, :] == cell_to_X_id[j])
                        )
                    )
                    if line_cell_to_X[0].size > 0:
                        au_cell_to_X[j] = AU[line_cell_to_X]
                    else:  # there is no line connecting original cell and cell at position X
                        au_cell_to_X[j] = np.array([0])
        return np.array(cell_to_X_coords), np.array(cell_to_X_centre), au_cell_to_X

    def subcell_info(
        self, index, cell_coords, cell_centre, subcell, subnb_indices, subnb_centres
    ):
        """ 
        Return:
            sub_coords: coordinates of subcell.
            tri_indices: indices of cells whose centres are vertices of triangle possibly used for interpolation.
            tri_vertices: vertices of triangle which can be used for interpolation.
        Args:
            index = index of cell.
            cell_coords: bottom-left and top-right coordinates of cell ([x_left, y_bottom, x_right, y_top])
            cell_centre: coordinates of centre of cell.
            subcell: 1 = left-bottom, 2 = right-bottom, 3 = right-top, 4 = left-top.
            subnb_indices (array): index of cell to left, bottom, right, top.
            subnb_centres (2x4 array): centres of cell to left, bottom, right, top.
        """
        tri_vertices = [subnb_centres[0], cell_centre, subnb_centres[1]]
        tri_indices = [subnb_indices[0], index, subnb_indices[1]]
        if subcell == 1:
            sub_coords = [
                cell_coords[0],
                cell_coords[1],
                cell_centre[0],
                cell_centre[1],
            ]
        elif subcell == 2:
            sub_coords = [
                cell_centre[0],
                cell_coords[1],
                cell_coords[2],
                cell_centre[1],
            ]
        elif subcell == 3:
            sub_coords = [
                cell_centre[0],
                cell_centre[1],
                cell_coords[2],
                cell_coords[3],
            ]
        else:
            sub_coords = [
                cell_coords[0],
                cell_centre[1],
                cell_centre[0],
                cell_coords[3],
            ]
        return sub_coords, tri_indices, tri_vertices

    def subsubcell_info(
        self,
        index,
        sub_centre,
        subsubnbs_indices,
        subsubnbs_centres,
        subsubnbs_au,
        subsubcell,
    ):
        """ 
        Return:
            subtri_vertices: vertices of triangle possibly used for interpolation.
            subtri_indices: indices of cells whose centres are vertices of triangle possibly used for interpolation.
            subau: wet cross-sectional area between neighbours and subsubcell.
        Args:
            index = index of cell.
            sub_centre: coordinates of centre of subcell.
            subsubnbs_indices: indices of 4 neighbours of subcell.
            subsubnbs_centres: coordinates of 4 centres of neighbours of subcell.
            subsubnbs_au: wet cross-sectional area between 4 neighbours and subcell.
            subsubcell: number of subsubcell.
        """
        if subsubcell == 1:
            subtri_vertices = [subsubnbs_centres[0], sub_centre, subsubnbs_centres[1]]
            subtri_indices = [subsubnbs_indices[0], index, subsubnbs_indices[1]]
            subau = [subsubnbs_au[0], subsubnbs_au[1]]
        elif subsubcell == 2:
            subtri_vertices = [subsubnbs_centres[1], sub_centre, subsubnbs_centres[2]]
            subtri_indices = [subsubnbs_indices[1], index, subsubnbs_indices[2]]
            subau = [subsubnbs_au[1], subsubnbs_au[2]]
        elif subsubcell == 3:
            subtri_vertices = [subsubnbs_centres[2], sub_centre, subsubnbs_centres[3]]
            subtri_indices = [subsubnbs_indices[2], index, subsubnbs_indices[3]]
            subau = [subsubnbs_au[2], subsubnbs_au[3]]
        else:
            subtri_vertices = [subsubnbs_centres[3], sub_centre, subsubnbs_centres[0]]
            subtri_indices = [subsubnbs_indices[3], index, subsubnbs_indices[0]]
            subau = [subsubnbs_au[3], subsubnbs_au[0]]
        return subtri_vertices, subtri_indices, subau

    def find_points_subcell(self, pts_in_cell, sub_coords, index_pts_in_cell):
        """ 
        Return:
            index_pts (array): indices of the points in the subcell.
            pts_in_cell[index_pts_in_sub] (array): points in the subcell.
        Args:
            pts_in_cell: array of points to be interpolated in subcell.
            sub_coords: bottom-left and top-right coordinates of subcell.
            index_pts_in_cell (array): indices of points located in this cell.
        """
        index_pts_in_sub = np.where(
            (pts_in_cell[:, 0] >= sub_coords[0])
            & (pts_in_cell[:, 0] <= sub_coords[2])
            & (pts_in_cell[:, 1] >= sub_coords[1])
            & (pts_in_cell[:, 1] <= sub_coords[3])
        )
        index_pts = index_pts_in_cell[index_pts_in_sub]
        return pts_in_cell[index_pts_in_sub], index_pts

    def X_neighbour(self, X, cell_coords, cell_centre, coords):
        """ 
        Return:
            index of neighbour(s) to the X of cell.
            type_nb: type of neighbour.
                0: neighbour has same size as cell.
                1: neighbour(s) is/are smaller than cell (possibly 2 neighbours).
                2: neighbour is larger than cell.
                3: no neighbour.
        Args:
            X: 'l' = left neighbour, 'b' = bottom neighbour, 'r' = right neighbour, 't' = top neighbour
            cell_coords: coordinates of cell.
            cell_centre: coordinates of centre of cell.
            coords (array): bottom-left and top-right coordinates of cells.
        """
        if X == 'l':
            index = np.where(
                (cell_coords[0] > coords[0, :])
                & (cell_coords[1] == coords[1, :])
                & (cell_coords[0] == coords[2, :])
                & (cell_coords[3] == coords[3, :])
            )
        elif X == 'r':
            index = np.where(
                (cell_coords[2] == coords[0, :])
                & (cell_coords[1] == coords[1, :])
                & (cell_coords[2] < coords[2, :])
                & (cell_coords[3] == coords[3, :])
            )
        elif X == 'b':
            index = np.where(
                (cell_coords[0] == coords[0, :])
                & (cell_coords[1] > coords[1, :])
                & (cell_coords[2] == coords[2, :])
                & (cell_coords[1] == coords[3, :])
            )
        else:
            index = np.where(
                (cell_coords[0] == coords[0, :])
                & (cell_coords[3] == coords[1, :])
                & (cell_coords[2] == coords[2, :])
                & (cell_coords[3] < coords[3, :])
            )
        if index[0].size == 0:  # search for smaller neighbours
            if X == 'l':  # search for lower/upper left
                index_small_1 = np.where(
                    (cell_coords[0] > coords[0, :])
                    & (cell_coords[1] == coords[1, :])
                    & (cell_coords[0] == coords[2, :])
                    & (cell_centre[1] == coords[3, :])
                )
                index_small_2 = np.where(
                    (cell_coords[0] > coords[0, :])
                    & (cell_centre[1] == coords[1, :])
                    & (cell_coords[0] == coords[2, :])
                    & (cell_coords[3] == coords[3, :])
                )
            elif X == 'r':  # search for lower/upper right
                index_small_1 = np.where(
                    (cell_coords[2] == coords[0, :])
                    & (cell_coords[1] == coords[1, :])
                    & (cell_coords[2] < coords[2, :])
                    & (cell_coords[3] > coords[3, :])
                )
                index_small_2 = np.where(
                    (cell_coords[2] == coords[0, :])
                    & (cell_coords[1] < coords[1, :])
                    & (cell_coords[2] < coords[2, :])
                    & (cell_coords[3] == coords[3, :])
                )
            elif X == 'b':  # search for lower left/right
                index_small_1 = np.where(
                    (cell_coords[0] == coords[0, :])
                    & (cell_coords[1] > coords[1, :])
                    & (cell_coords[2] > coords[2, :])
                    & (cell_coords[1] == coords[3, :])
                )
                index_small_2 = np.where(
                    (cell_coords[0] < coords[0, :])
                    & (cell_coords[1] > coords[1, :])
                    & (cell_coords[2] == coords[2, :])
                    & (cell_coords[1] == coords[3, :])
                )
            else:  # search for upper left/right
                index_small_1 = np.where(
                    (cell_coords[0] == coords[0, :])
                    & (cell_coords[3] == coords[1, :])
                    & (cell_coords[2] > coords[2, :])
                    & (cell_coords[3] < coords[3, :])
                )
                index_small_2 = np.where(
                    (cell_centre[0] == coords[0, :])
                    & (cell_coords[3] == coords[1, :])
                    & (cell_coords[2] == coords[2, :])
                    & (cell_coords[3] < coords[3, :])
                )
            if not (index_small_1[0].size == 0) or not (index_small_2[0].size == 0):
                type_nb = 1  # smaller neighbour exists
                if not (index_small_1[0].size == 0) and not (
                    index_small_2[0].size == 0
                ):
                    return np.array([index_small_1[0][0], index_small_2[0][0]]), type_nb
                elif not (index_small_1[0].size == 0):
                    return np.array(index_small_1[0][0]), type_nb
                else:
                    return np.array(index_small_2[0][0]), type_nb
            else:  # search for larger neighbours
                if X == 'l':  # lower/upper large cell
                    index_large_1 = np.where(
                        (cell_coords[0] > coords[0, :])
                        & (cell_coords[1] > coords[1, :])
                        & (cell_coords[0] == coords[2, :])
                        & (cell_coords[3] == coords[3, :])
                    )
                    index_large_2 = np.where(
                        (cell_coords[0] > coords[0, :])
                        & (cell_coords[1] == coords[1, :])
                        & (cell_coords[0] == coords[2, :])
                        & (cell_coords[3] < coords[3, :])
                    )
                elif X == 'r':  # lower/upper large cell
                    index_large_1 = np.where(
                        (cell_coords[2] == coords[0, :])
                        & (cell_coords[1] > coords[1, :])
                        & (cell_coords[2] < coords[2, :])
                        & (cell_coords[3] == coords[3, :])
                    )
                    index_large_2 = np.where(
                        (cell_coords[2] == coords[0, :])
                        & (cell_coords[1] == coords[1, :])
                        & (cell_coords[2] < coords[2, :])
                        & (cell_coords[3] < coords[3, :])
                    )
                elif X == 'b':  # left/right large cell
                    index_large_1 = np.where(
                        (cell_coords[0] > coords[0, :])
                        & (cell_coords[1] > coords[1, :])
                        & (cell_coords[2] == coords[2, :])
                        & (cell_coords[1] == coords[3, :])
                    )
                    index_large_2 = np.where(
                        (cell_coords[0] == coords[0, :])
                        & (cell_coords[1] > coords[1, :])
                        & (cell_coords[2] < coords[2, :])
                        & (cell_coords[1] == coords[3, :])
                    )
                else:  # left/right large cell
                    index_large_1 = np.where(
                        (cell_coords[0] > coords[0, :])
                        & (cell_coords[3] == coords[1, :])
                        & (cell_coords[2] == coords[2, :])
                        & (cell_coords[3] < coords[3, :])
                    )
                    index_large_2 = np.where(
                        (cell_coords[0] == coords[0, :])
                        & (cell_coords[3] == coords[1, :])
                        & (cell_coords[2] < coords[2, :])
                        & (cell_coords[3] < coords[3, :])
                    )
                if not (index_large_1[0].size == 0):
                    type_nb = 2  # larger neighbour exists
                    return np.array(index_large_1[0][0], dtype='int64'), type_nb
                elif not (index_large_2[0].size == 0):
                    type_nb = 2
                    return np.array(index_large_2[0][0], dtype='int64'), type_nb
                else:
                    type_nb = 3  # no neighbour
                    return np.array([[]]), type_nb
        else:
            type_nb = 0  # same size neighbour exists
            return np.array(index[0][0]), type_nb

    def find_neighbours_subcell(
        self, subcell, cell_centre, nb_type, nb_indices, nb_coords, nb_centres, nb_au
    ):
        """ 
        Return indices, coordinates, centres, wet cross-sectional area and type of neighbours of subcell.
        Args:
            subcell: 1 = left-bottom, 2 = right-bottom, 3 = right-top, 4 = left-top.
            cell_centre: coordinates of centre of cell.
            nb_type: types of neighbours to left, bottom, right, top (0 = same size, 1 = smaller, 2 = larger, 3 = nonexistent).
            nb_indices: indices of neighbours to left, bottom, right, top.
            nb_coords: bottom-left and top-right coordinates of neighbours.
            nb_centres: coordinates of centres of neighbours.
            nb_au: wet cross-sectional area between cell and neighbours.
        """
        kwargs_assist = dict(subcell=subcell, cell_centre=cell_centre, nb_type=nb_type)
        if subcell == 1:
            index_l, coords_l, centre_l, au_l, type_l = self.find_neighbours_subcell_assist(
                'l',
                nb_indices[0],
                nb_coords[0],
                nb_centres[0],
                nb_au[0],
                **kwargs_assist
            )
            index_b, coords_b, centre_b, au_b, type_b = self.find_neighbours_subcell_assist(
                'b',
                nb_indices[1],
                nb_coords[1],
                nb_centres[1],
                nb_au[1],
                **kwargs_assist
            )
            return (
                [index_l, index_b],
                [coords_l, coords_b],
                [centre_l, centre_b],
                [au_l, au_b],
                np.array([type_l, type_b]),
            )
        elif subcell == 2:
            index_b, coords_b, centre_b, au_b, type_b = self.find_neighbours_subcell_assist(
                'b',
                nb_indices[1],
                nb_coords[1],
                nb_centres[1],
                nb_au[1],
                **kwargs_assist
            )
            index_r, coords_r, centre_r, au_r, type_r = self.find_neighbours_subcell_assist(
                'r',
                nb_indices[2],
                nb_coords[2],
                nb_centres[2],
                nb_au[2],
                **kwargs_assist
            )
            return (
                [index_b, index_r],
                [coords_b, coords_r],
                [centre_b, centre_r],
                [au_b, au_r],
                np.array([type_b, type_r]),
            )
        elif subcell == 3:
            index_r, coords_r, centre_r, au_r, type_r = self.find_neighbours_subcell_assist(
                'r',
                nb_indices[2],
                nb_coords[2],
                nb_centres[2],
                nb_au[2],
                **kwargs_assist
            )
            index_t, coords_t, centre_t, au_t, type_t = self.find_neighbours_subcell_assist(
                't',
                nb_indices[3],
                nb_coords[3],
                nb_centres[3],
                nb_au[3],
                **kwargs_assist
            )
            return (
                [index_r, index_t],
                [coords_r, coords_t],
                [centre_r, centre_t],
                [au_r, au_t],
                np.array([type_r, type_t]),
            )
        else:
            index_t, coords_t, centre_t, au_t, type_t = self.find_neighbours_subcell_assist(
                't',
                nb_indices[3],
                nb_coords[3],
                nb_centres[3],
                nb_au[3],
                **kwargs_assist
            )
            index_l, coords_l, centre_l, au_l, type_l = self.find_neighbours_subcell_assist(
                'l',
                nb_indices[0],
                nb_coords[0],
                nb_centres[0],
                nb_au[0],
                **kwargs_assist
            )
            return (
                [index_t, index_l],
                [coords_t, coords_l],
                [centre_t, centre_l],
                [au_t, au_l],
                np.array([type_t, type_l]),
            )

    def find_neighbours_subcell_assist(
        self,
        X,
        indexX,
        cell_to_X_coords,
        cell_to_X_centre,
        au_cell_to_X,
        subcell,
        cell_centre,
        nb_type,
    ):
        """ 
        Return indices, coordinates, centres, wet cross-sectional area and type of neighbour at location X of subcell
            ('l' = left, 'b' = bottom, 'r' = right, 't' = top).
        """
        if X == 'l':
            subcell1 = 1
            subcell2 = 4
            n_t_index = 0
            row = 1
        elif X == 'r':
            subcell1 = 2
            subcell2 = 3
            n_t_index = 2
            row = 1
        elif X == 'b':
            subcell1 = 1
            subcell2 = 2
            n_t_index = 1
            row = 0
        elif X == 't':
            subcell1 = 4
            subcell2 = 3
            n_t_index = 3
            row = 0

        if indexX.size > 1:  # cell has two smaller neighbours
            if subcell == subcell1 or subcell == subcell2:
                if subcell == subcell1:
                    location = np.where(
                        cell_to_X_centre[row, :] == np.min(cell_to_X_centre[row, :])
                    )[0][0]
                else:
                    location = np.where(
                        cell_to_X_centre[row, :] == np.max(cell_to_X_centre[row, :])
                    )[0][0]
                index_X = indexX[location]
                single_cell_to_X_centre = cell_to_X_centre[:, location]
                single_au_cell_to_X = au_cell_to_X[location]
                single_cell_to_X_coords = cell_to_X_coords[:, location]
                type_X = nb_type[n_t_index]
        else:
            if nb_type[n_t_index] == 1:  # cell has 1 smaller neighbour
                if (
                    subcell == subcell1 and cell_to_X_centre[row] <= cell_centre[row]
                ) or (
                    subcell == subcell2 and cell_to_X_centre[row] >= cell_centre[row]
                ):
                    index_X = indexX
                    single_cell_to_X_centre = cell_to_X_centre
                    single_au_cell_to_X = au_cell_to_X
                    single_cell_to_X_coords = cell_to_X_coords
                    type_X = nb_type[n_t_index]
                else:
                    index_X = np.array([])
                    single_cell_to_X_centre = [self.no_data_value, self.no_data_value]
                    single_au_cell_to_X = 0
                    single_cell_to_X_coords = []
                    type_X = 3
            else:  # cell has same size/larger/no neighbour
                index_X = indexX
                single_cell_to_X_centre = cell_to_X_centre
                single_au_cell_to_X = au_cell_to_X
                single_cell_to_X_coords = cell_to_X_coords
                type_X = nb_type[n_t_index]
        return (
            index_X,
            single_cell_to_X_coords,
            single_cell_to_X_centre,
            single_au_cell_to_X,
            type_X,
        )

    def average_x(self, pts_in_sub, Matrix, water):
        """ 
        Return interpolated (array): waterlevels averaged based on distance in x-coordinates.
        Args:
            pts_in_sub: points in subcell.
            Matrix: matrix with 1st row x-coords of cell and its neighbour, 2nd row y-coords of cell and its neighbour, 3rd row ones.
            water:  waterlevels of cell and its neighbour.
        """
        interpolated = np.zeros(len(pts_in_sub))
        difx = abs(Matrix[0][0] - Matrix[0][1])
        weight1 = 1 - abs(pts_in_sub[:, 0] - Matrix[0][0]) / difx
        weight2 = 1 - abs(pts_in_sub[:, 0] - Matrix[0][1]) / difx
        interpolated = weight1 * water[0] + weight2 * water[1]
        return interpolated

    def average_y(self, pts_in_sub, Matrix, water):
        """ 
        Return interpolated (array): waterlevels averaged based on distance in y-coordinates.
        Args:
            pts_in_sub: points in subcell.
            Matrix: matrix with 1st row x-coords of cell and its neighbour, 2nd row y-coords of cell and its neighbour, 3rd row ones.
            water:  waterlevels of cell and its neighbour.
        """
        interpolated = np.zeros(len(pts_in_sub))
        dify = abs(Matrix[1][0] - Matrix[1][1])
        weight1 = 1 - abs(pts_in_sub[:, 1] - Matrix[1][0]) / dify
        weight2 = 1 - abs(pts_in_sub[:, 1] - Matrix[1][1]) / dify
        interpolated = weight1 * water[0] + weight2 * water[1]
        return interpolated

    def find_closest(self, centres, pt):
        """ 
        Return centre that is closest to pt.
        """
        centres = np.array(centres)
        pt = np.array(pt)
        distance = np.sqrt((centres[:, 0] - pt[0]) ** 2 + (centres[:, 1] - pt[1]) ** 2)
        location_min = np.where(distance == np.min(distance))
        return centres[location_min[-1], :][0]

    def refine_cell(self, cell_coords, cell_centre):
        """ 
        Return coordinates of left-bottom and right-top corners and of centres of refined cells (the cell is split into 4 smaller cells).
        Args:
            cell_coords: coordinates of left-bottom and right-top corner.
            cell_centre: coordinates of centre of cell.
        """
        # left_bottom:
        lb_coords = [cell_coords[0], cell_coords[1], cell_centre[0], cell_centre[1]]
        lb_centre = [
            lb_coords[0] + (lb_coords[2] - lb_coords[0]) / 2,
            lb_coords[1] + (lb_coords[3] - lb_coords[1]) / 2,
        ]

        # right_bottom:
        rb_coords = [cell_centre[0], cell_coords[1], cell_coords[2], cell_centre[1]]
        rb_centre = [
            rb_coords[0] + (rb_coords[2] - rb_coords[0]) / 2,
            rb_coords[1] + (rb_coords[3] - rb_coords[1]) / 2,
        ]

        # right_top:
        rt_coords = [cell_centre[0], cell_centre[1], cell_coords[2], cell_coords[3]]
        rt_centre = [
            rt_coords[0] + (rt_coords[2] - rt_coords[0]) / 2,
            rt_coords[1] + (rt_coords[3] - rt_coords[1]) / 2,
        ]

        # left_top:
        lt_coords = [cell_coords[0], cell_centre[1], cell_centre[0], cell_coords[3]]
        lt_centre = [
            lt_coords[0] + (lt_coords[2] - lt_coords[0]) / 2,
            lt_coords[1] + (lt_coords[3] - lt_coords[1]) / 2,
        ]

        fine_centres = [lb_centre, rb_centre, rt_centre, lt_centre]
        fine_coords = [lb_coords, rb_coords, rt_coords, lt_coords]
        return fine_coords, fine_centres

    def find_containing_cell(self, fine_coords, centre):
        """ 
        Return coordinates of the refined cell containing centre.
        """
        fine_coords = np.array(fine_coords)
        centre = np.array(centre)
        return fine_coords[
            np.where(
                (centre[0] > fine_coords[:, 0])
                & (centre[0] < fine_coords[:, 2])
                & (centre[1] > fine_coords[:, 1])
                & (centre[1] < fine_coords[:, 3])
            ),
            :,
        ][0][0]

    def bary_lin_interpolator(
        self, pts_in_sub, tri_indices, modified_Matrix, waterlevels
    ):
        """ 
        Return interpolated water levels via linear barycentric interpolation.
        """
        modified_Matrix = np.matrix(modified_Matrix, dtype='float')
        C = np.array(
            [
                waterlevels[tri_indices[0]],
                waterlevels[tri_indices[1]],
                waterlevels[tri_indices[2]],
            ],
            dtype=np.ndarray,
        )
        grid = np.c_[pts_in_sub, np.ones(len(pts_in_sub))].transpose()
        bary = np.dot(np.linalg.inv(modified_Matrix), grid)
        return bary.transpose() @ C

    def triangle_IDSW_interpolator(
        self, pts_in_sub, waterlevels, tri_indices, tri_vertices
    ):
        """ 
        Return interpolated: array of IDSW (on triangles) interpolated water levels at points in subcell.
        Args:
            pts_in_sub: array of points to be interpolated in subcell.
            waterlevels: array of waterlevels (s1).
            tri_indices: indices of cells whose centres are vertices of triangle used for interpolation.
            tri_vertices: vertices of triangle used ofr interpolation.
        """

        def distance(pts, pt):  # computes distance between array of points and a point
            return np.sqrt((pts[:, 0] - pt[0]) ** 2 + (pts[:, 1] - pt[1]) ** 2)

        p = 2
        C = np.array(
            [
                waterlevels[tri_indices[0]],
                waterlevels[tri_indices[1]],
                waterlevels[tri_indices[2]],
            ],
            dtype=np.ndarray,
        )
        interpolated = np.zeros(len(pts_in_sub))
        frac1 = np.zeros(len(pts_in_sub))
        frac2 = np.zeros(len(pts_in_sub))
        frac3 = np.zeros(len(pts_in_sub))
        ind = []
        dist_pt_1 = distance(pts_in_sub, tri_vertices[0][:])
        dist_pt_2 = distance(pts_in_sub, tri_vertices[1][:])
        dist_pt_3 = distance(pts_in_sub, tri_vertices[2][:])
        # avoid division by 0:
        if np.any(dist_pt_1 == 0):
            ind1 = np.where(dist_pt_1 == 0)
            ind = ind + ind1
            frac1[ind1] = 1
        if np.any(dist_pt_2 == 0):
            ind2 = np.where(dist_pt_2 == 0)
            ind = ind + ind2
            frac2[ind2] = 1
        if np.any(dist_pt_3 == 0):
            ind3 = np.where(dist_pt_3 == 0)
            ind = ind + ind3
            frac3[ind3] = 1
        index = np.ones(len(pts_in_sub), bool)
        index[ind] = False

        frac1[index] = 1 / dist_pt_1[index] ** p
        frac2[index] = 1 / dist_pt_2[index] ** p
        frac3[index] = 1 / dist_pt_3[index] ** p
        total_frac = frac1 + frac2 + frac3
        weight1 = frac1 / total_frac
        weight2 = frac2 / total_frac
        weight3 = frac3 / total_frac
        interpolated = weight1 * C[0] + weight2 * C[1] + weight3 * C[2]
        return interpolated


class Improvement1(MyCalculator):
    # Interpolation via barycentric or IDSW interpolation on triangles.

    def compute_nr_neighbours(self, index, waterlevels, au, Matrix):
        """ 
        Return:
            how many neighbours cell has that can be used for interpolation.
            Matrix modified for barycentric interpolation.
            water gives waterlevels of cell and its neighbour when interpolation_type == 2.
        Args:
            index: index of cell.
            waterlevels (array): water levels of cells.
            au: wet cross-sectional area between cell and neighbours.
            Matrix: matrix with 1st row x-coords of cell and its neighbours, 2nd row y-coords of cell and its neighbours, 3rd row ones.
        """
        # initialize
        nbs = [1, 1, 1]
        water = [-1, -1]
        i = -1
        k = 0

        for j in range(3):
            i = i + 1
            if (
                (
                    (not (j == 1))
                    and (
                        (index[j].size == 0)
                        or (waterlevels[index[j]] == self.no_data_value)
                    )
                )
                or ((j == 0) and (au[0] < 10 ** (-7)))
                or ((j == 2) and (au[1] < 10 ** (-7)))
            ):
                # no neighbour-cell with known waterlevels or waterflow
                Matrix = np.delete(Matrix, i, 1)
                i = i - 1
                nbs[j] = 0
            elif k < 2:
                water[k] = waterlevels[index[j]]
                k = k + 1
        return np.sum(nbs) - 1, Matrix, water

    def determine_interpolation_type(self, tri_indices, waterlevels, au, Matrix):
        """ 
        Return:
            interpolation_type: type to be used based on the number and type of neighbours
                (0 = error, cell has no known waterlevel, should not interpolate,
                 1 = no neighbours, use waterlevel of cell
                 2 = 1 neighbour, use average waterlevel based on distance in x- or y-direction
                 3 = 2 neighbours, use barycentric or IDSW interpolation.
            water: waterlevels of cell and its neighbour when interpolation_type == 2.
            modified_Matrix: Matrix modified for barycentric-interpolation when interpolation_type == 3.
        """
        if waterlevels[tri_indices[1]] == self.no_data_value:
            interpolation_type = 0
            print("error: interpolation type 0")
        nr_nbs, modified_Matrix, water = self.compute_nr_neighbours(
            tri_indices, waterlevels, au, Matrix
        )
        if nr_nbs == 0:
            interpolation_type = 1
        elif nr_nbs == 1:
            interpolation_type = 2
        else:
            interpolation_type = 3
        return interpolation_type, water, modified_Matrix

    def pseudo_interpolation(
        self,
        index,
        cell_centre,
        cell_coords,
        tri_indices,
        tri_vertices,
        subcell,
        waterlevels,
        pts_in_sub,
    ):
        """ 
        Return array of interpolated water levels computed by alternative way for when only 1 neighbour by creating new neighbour same waterlevel as cell.
        """
        pseudo_water = np.full(3, waterlevels[index])
        pseudo_coords = tri_vertices.copy()
        dx = 2 * (cell_centre[0] - cell_coords[0])
        dy = 2 * (cell_centre[1] - cell_coords[1])
        if subcell == 1:
            for j in range(3):
                if tri_indices[j].size > 0:
                    pseudo_water[j] = waterlevels[tri_indices[j]]
                else:
                    if j == 0:
                        pseudo_coords[j] = [cell_centre[0] - dx, cell_centre[1]]
                    if j == 2:
                        pseudo_coords[j] = [cell_centre[0], cell_centre[1] - dy]
        elif subcell == 2:
            for j in range(3):
                if tri_indices[j].size > 0:
                    pseudo_water[j] = waterlevels[tri_indices[j]]
                else:
                    if j == 0:
                        pseudo_coords[j] = [cell_centre[0], cell_centre[1] - dy]
                    if j == 2:
                        pseudo_coords[j] = [cell_centre[0] + dx, cell_centre[1]]
        elif subcell == 3:
            for j in range(3):
                if tri_indices[j].size > 0:
                    pseudo_water[j] = waterlevels[tri_indices[j]]
                else:
                    if j == 0:
                        pseudo_coords[j] = [cell_centre[0] + dx, cell_centre[1]]
                    if j == 2:
                        pseudo_coords[j] = [cell_centre[0], cell_centre[1] + dy]
        elif subcell == 4:
            for j in range(3):
                if tri_indices[j].size > 0:
                    pseudo_water[j] = waterlevels[tri_indices[j]]
                else:
                    if j == 0:
                        pseudo_coords[j] = [cell_centre[0], cell_centre[1] + dy]
                    if j == 2:
                        pseudo_coords[j] = [cell_centre[0] - dx, cell_centre[1]]
        Matrix = [
            [pseudo_coords[0][0], pseudo_coords[1][0], pseudo_coords[2][0]],
            [pseudo_coords[0][1], pseudo_coords[1][1], pseudo_coords[2][1]],
            [1, 1, 1],
        ]
        modified_Matrix = np.matrix(Matrix, dtype='float')
        C = np.array(pseudo_water, dtype=np.ndarray)
        grid = np.c_[pts_in_sub, np.ones(len(pts_in_sub))].transpose()
        bary = np.dot(np.linalg.inv(modified_Matrix), grid)
        return bary.transpose() @ C

    def Central_Interpolator(
        self, pts_in_sub, waterlevels, tri_indices, tri_vertices, au, Matrix
    ):
        """ 
        Return interpolated: array of interpolated waterlevels at points in subcell, based on barycentric linear or IDSW interpolation.
        Args:
            pts_in_sub: array of points to be interpolated in subcell.
            waterlevels: array of waterlevels (s1).
            tri_indices: indices of cells whose centres are vertices of triangle used for interpolation.
            tri_vertices: vertices of triangle used for interpolation.
            au: array of wet cross-sectional area of two neighbour-cells.
            Matrix: matrix with 1st row x-coords, 2nd row y-coords, 3rd row ones.
        """
        nr_points = len(pts_in_sub)
        interpolation_type, water, modified_Matrix = self.determine_interpolation_type(
            tri_indices, waterlevels, au, Matrix
        )
        if interpolation_type == 1:
            interpolated = waterlevels[tri_indices[1]] * np.ones(nr_points)
            return interpolated
        elif interpolation_type == 2:
            if not (
                modified_Matrix[0][0] - modified_Matrix[0][1] == 0
            ):  # neighbour is to left or right (different x-, same y-coord.s)
                interpolated = self.average_x(pts_in_sub, modified_Matrix, water)
                return interpolated
            elif not (
                modified_Matrix[1][0] - modified_Matrix[1][1] == 0
            ):  # neighbour is to bottom or top (different y-, same x-coord.s)
                interpolated = self.average_y(pts_in_sub, modified_Matrix, water)
                return interpolated
        else:
            if INTERPOLATION_METHOD == 'barycentric':
                interpolated = self.bary_lin_interpolator(
                    pts_in_sub, tri_indices, modified_Matrix, waterlevels
                )
            elif INTERPOLATION_METHOD == 'distance':
                interpolated = self.triangle_IDSW_interpolator(
                    pts_in_sub, waterlevels, tri_indices, tri_vertices
                )
            return interpolated

    def refine_neighbours(self, subnb_types, subnb_coords, subnb_centres):
        """ 
        Return type, coordinates and centres of refined cells of original cell that are also neighbours of the subcell.
        Args:
            subnb_types: type of neighbours of subcell.
            subnb_coords: coordinates of neighbours of subcell.
            subnb_centres: coordinates of centres of neighbours of subcell.
        """
        if subnb_types[0] == 1 or subnb_types[0] == 3:  # smaller or nonexistent
            centre_nb1 = subnb_centres[0]
        if subnb_types[1] == 1 or subnb_types[1] == 3:  # smaller or nonexistent
            centre_nb2 = subnb_centres[1]
        if subnb_types[0] == 0:  # same size, refine once
            nb1_fine_coords, nb1_fine_centres = self.refine_cell(
                subnb_coords[0], subnb_centres[0]
            )
            centre_nb1 = self.find_closest(nb1_fine_centres, centre_nb2)
        if subnb_types[1] == 0:  # same size, refine once
            nb2_fine_coords, nb2_fine_centres = self.refine_cell(
                subnb_coords[1], subnb_centres[1]
            )
            centre_nb2 = self.find_closest(nb2_fine_centres, centre_nb1)
        if subnb_types[0] == 2:  # larger, refine twice
            nb1_fine_coords1, nb1_fine_centres1 = self.refine_cell(
                subnb_coords[0], subnb_centres[0]
            )
            centre_nb11 = self.find_closest(nb1_fine_centres1, centre_nb2)
            coords_nb11 = self.find_containing_cell(nb1_fine_coords1, centre_nb11)
            nb1_fine_coords, nb1_fine_centres = self.refine_cell(
                coords_nb11, centre_nb11
            )
            centre_nb1 = self.find_closest(nb1_fine_centres, centre_nb2)
        if subnb_types[1] == 2:  # larger, refine twice
            nb2_fine_coords1, nb2_fine_centres1 = self.refine_cell(
                subnb_coords[1], subnb_centres[1]
            )
            centre_nb21 = self.find_closest(nb2_fine_centres1, centre_nb1)
            coords_nb21 = self.find_containing_cell(nb2_fine_coords1, centre_nb21)
            nb2_fine_coords, nb2_fine_centres = self.refine_cell(
                coords_nb21, centre_nb21
            )
            centre_nb2 = self.find_closest(nb2_fine_centres, centre_nb1)
        return centre_nb1, centre_nb2

    def small_neighbour_interpolator(
        self,
        index,
        cell_coords,
        cell_centre,
        sub_coords,
        sub_centre,
        waterlevels,
        subnb_indices,
        subnb_coords,
        subnb_centres,
        subnb_au,
        subcell,
        subnb_types,
        pts_in_sub,
        index_pts_in_sub,
        result,
    ):
        """ 
        Return result, containing interpolated values at indices of points in subcell, computed by linear barycentric interpolation or IDSW interpolation
        """
        # One of neighbours is small. Cell is refined and if necessary neighbours are refined. Subcell is refined.
        fine_coords, fine_centres = self.refine_cell(cell_coords, cell_centre)
        centre_nb1, centre_nb2 = self.refine_neighbours(
            subnb_types, subnb_coords, subnb_centres
        )
        sub_fine_coords, sub_fine_centres = self.refine_cell(sub_coords, sub_centre)

        if subcell == 1:
            subsubnbs_indices = [subnb_indices[0], subnb_indices[1], index, index]
            subsubnbs_centres = [
                centre_nb1,
                centre_nb2,
                fine_centres[1],
                fine_centres[3],
            ]
            subsubnbs_au = [subnb_au[0], subnb_au[1], 1, 1]
        elif subcell == 2:
            subsubnbs_indices = [index, subnb_indices[0], subnb_indices[1], index]
            subsubnbs_centres = [
                fine_centres[0],
                centre_nb1,
                centre_nb2,
                fine_centres[2],
            ]
            subsubnbs_au = [1, subnb_au[0], subnb_au[1], 1]
        elif subcell == 3:
            subsubnbs_indices = [index, index, subnb_indices[0], subnb_indices[1]]
            subsubnbs_centres = [
                fine_centres[3],
                fine_centres[1],
                centre_nb1,
                centre_nb2,
            ]
            subsubnbs_au = [1, 1, subnb_au[0], subnb_au[1]]
        else:
            subsubnbs_indices = [subnb_indices[1], index, index, subnb_indices[0]]
            subsubnbs_centres = [
                centre_nb2,
                fine_centres[0],
                fine_centres[2],
                centre_nb1,
            ]
            subsubnbs_au = [subnb_au[1], 1, 1, subnb_au[0]]

        for subsubcell in range(1, 5):
            subsub_coords = sub_fine_coords[subsubcell - 1]
            subtri_vertices, subtri_indices, subau = self.subsubcell_info(
                index,
                sub_centre,
                subsubnbs_indices,
                subsubnbs_centres,
                subsubnbs_au,
                subsubcell,
            )
            pts_in_subsub, subindex_pts = self.find_points_subcell(
                pts_in_sub, subsub_coords, index_pts_in_sub
            )
            if pts_in_subsub.size == 0:
                continue
            subMatrix = [
                [subtri_vertices[0][0], subtri_vertices[1][0], subtri_vertices[2][0]],
                [subtri_vertices[0][1], subtri_vertices[1][1], subtri_vertices[2][1]],
                [1, 1, 1],
            ]
            subinterpolated = self.Central_Interpolator(
                pts_in_subsub,
                waterlevels,
                subtri_indices,
                subtri_vertices,
                subau,
                subMatrix,
            )
            result[subindex_pts] = subinterpolated
        return result

    def large_neighbour_interpolator(
        self,
        index,
        cell_centre,
        pts_in_sub,
        subnb_types,
        subnb_indices,
        subnb_coords,
        subnb_centres,
        waterlevels,
        au,
    ):
        """ 
        Return interpolated: array of interpolated waterlevels at points in subcell, computed by linear barycentric interpolation or IDSW interpolation.
        """
        # One of neighbours is large, no neighbours are small.
        if subnb_types[0] == 2:  # large, refine once
            nb1_fine_coords, nb1_fine_centres = self.refine_cell(
                subnb_coords[0], subnb_centres[0]
            )
            centre_nb1 = self.find_closest(nb1_fine_centres, cell_centre)
        else:  # same size or nonexistent
            centre_nb1 = subnb_centres[0]
        if subnb_types[1] == 2:  # large, refine once
            nb2_fine_coords, nb2_fine_centres = self.refine_cell(
                subnb_coords[1], subnb_centres[1]
            )
            centre_nb2 = self.find_closest(nb2_fine_centres, cell_centre)
        else:  # same size or nonexistent
            centre_nb2 = subnb_centres[1]
        tri_vertices = [centre_nb1, cell_centre, centre_nb2]
        tri_indices = np.array(
            [np.array([subnb_indices[0]]), index, np.array([subnb_indices[1]])],
            dtype=np.ndarray,
        )
        Matrix = [
            [tri_vertices[0][0], tri_vertices[1][0], tri_vertices[2][0]],
            [tri_vertices[0][1], tri_vertices[1][1], tri_vertices[2][1]],
            [1, 1, 1],
        ]
        interpolated = self.Central_Interpolator(
            pts_in_sub, waterlevels, tri_indices, tri_vertices, au, Matrix
        )
        return interpolated

    def __call__(self, indices, values, no_data_value):
        """ 
        Return result containing interpolated waterlevels.
            Interpolation done by linear barycentric interpolation if INTERPOLATION_METHOD == 'barycentric'
            Interpolation done by IDSW triangles if INTERPOLATION_METHOD == 'distance'
        Args:
            indices: indicies = (yoff, xoff), (yoff + ysize, xoff + xsize) in partition.
            values: result needs to have same shape.
            no_data_value: -9999
        """
        print(indices)

        # interpolating points
        points = self._get_points(indices)
        nr_points = len(points)
        pt_to_cell = self._get_nodgrid(indices).reshape(
            nr_points
        )  # array giving for each point in which cell it is.

        # initialize result
        self.no_data_value = no_data_value
        result = np.full(nr_points, self.no_data_value)

        max_nodes, max_lines = self.nr_nodes_lines
        # info nodes/cells
        x = np.arange(1, max_nodes + 1).reshape(1, -1)
        nodes_id_s1_coords = self.lookup_nodes[x][0]
        centres = nodes_id_s1_coords[:, 1:3]  # cell-centre coordinates
        waterlevels = nodes_id_s1_coords[:, 0]  # waterlevels in 2D-open-water cells
        coords = nodes_id_s1_coords[
            :, 3:7
        ].transpose()  # coords of cell left-bottom corner and right-top corner

        # info lines
        x_lines = np.arange(0, max_lines).reshape(1, -1)
        lines_line_linecoords_au = self.lookup_lines[x_lines][0]
        lines_wrt_cells = lines_line_linecoords_au[
            :, 0:2
        ].transpose()  # 2D array of cell-ids that the line connects
        AU = lines_line_linecoords_au[:, 6].transpose()  # wet cross-sectional area

        # find in which cells interpolating points are contained
        indexrange = np.unique(pt_to_cell)
        indexrange = indexrange[indexrange != 0]

        # loop over cells that contain interpolating points
        for cell_id in indexrange:
            # info cell
            index = cell_id - 1
            cell_coords = coords[:, index]
            cell_centre_true = [centres[index, 0], centres[index, 1]]
            cell_centre = cell_centre_true.copy()

            # info points in cell
            index_pts_in_cell = np.where(pt_to_cell == cell_id)[0]
            pts_in_cell = points[index_pts_in_cell]

            if (
                waterlevels[index] == no_data_value
            ):  # check if is waterlevel available for cell
                result[index_pts_in_cell] = no_data_value
                continue

            kwargs_neighbour = dict(
                cell_coords=cell_coords, cell_centre=cell_centre, coords=coords
            )
            kwargs_info = dict(
                coords=coords,
                centres=centres,
                lines_wrt_cells=lines_wrt_cells,
                AU=AU,
                cell_id=cell_id,
            )
            # cell to left
            index_left, type_left = self.X_neighbour('l', **kwargs_neighbour)
            coords_left, centre_left, au_left = self.cell_info(
                index_left, **kwargs_info
            )

            # cell to right
            index_right, type_right = self.X_neighbour('r', **kwargs_neighbour)
            coords_right, centre_right, au_right = self.cell_info(
                index_right, **kwargs_info
            )

            # cell to bottom
            index_bottom, type_bottom = self.X_neighbour('b', **kwargs_neighbour)
            coords_bottom, centre_bottom, au_bottom = self.cell_info(
                index_bottom, **kwargs_info
            )

            # cell to top
            index_top, type_top = self.X_neighbour('t', **kwargs_neighbour)
            coords_top, cell_to_top_centre, au_top = self.cell_info(
                index_top, **kwargs_info
            )

            nb_indices = [index_left, index_bottom, index_right, index_top]
            nb_coords = [coords_left, coords_bottom, coords_right, coords_top]
            nb_centres = [
                centre_left.copy(),
                centre_bottom.copy(),
                centre_right.copy(),
                cell_to_top_centre.copy(),
            ]
            nb_au = [au_left, au_bottom, au_right, au_top]
            nb_type = np.array([type_left, type_bottom, type_right, type_top])

            # loop over subcell/quadrant of each cell
            for subcell in range(1, 5):
                cell_centre = cell_centre_true.copy()
                # info subcell
                subnb_indices, subnb_coords, subnb_centres, subnb_au, subnb_types = self.find_neighbours_subcell(
                    subcell,
                    cell_centre,
                    nb_type,
                    nb_indices,
                    nb_coords,
                    nb_centres,
                    nb_au,
                )
                sub_coords, tri_indices, tri_vertices = self.subcell_info(
                    index,
                    cell_coords,
                    cell_centre,
                    subcell,
                    subnb_indices,
                    subnb_centres,
                )
                sub_centre = [
                    sub_coords[0] + (sub_coords[2] - sub_coords[0]) / 2,
                    sub_coords[1] + (sub_coords[3] - sub_coords[1]) / 2,
                ]

                # find interpolating points in each subcel
                pts_in_sub, index_pts = self.find_points_subcell(
                    pts_in_cell, sub_coords, index_pts_in_cell
                )
                if pts_in_sub.size == 0:  # no points in subcell
                    continue

                if np.any(subnb_types == 1):  # at least 1 neighbour smaller
                    result = self.small_neighbour_interpolator(
                        index,
                        cell_coords,
                        cell_centre,
                        sub_coords,
                        sub_centre,
                        waterlevels,
                        subnb_indices,
                        subnb_coords,
                        subnb_centres,
                        subnb_au,
                        subcell,
                        subnb_types,
                        pts_in_sub,
                        index_pts,
                        result,
                    )
                elif np.any(subnb_types == 2):  # at least 1 neighbour larger
                    interpolated = self.large_neighbour_interpolator(
                        index,
                        cell_centre,
                        pts_in_sub,
                        subnb_types,
                        subnb_indices,
                        subnb_coords,
                        subnb_centres,
                        waterlevels,
                        subnb_au,
                    )
                    result[index_pts] = interpolated
                else:  # all same size or no neighbours
                    Matrix = [
                        [tri_vertices[0][0], tri_vertices[1][0], tri_vertices[2][0]],
                        [tri_vertices[0][1], tri_vertices[1][1], tri_vertices[2][1]],
                        [1, 1, 1],
                    ]
                    interpolated = self.Central_Interpolator(
                        pts_in_sub,
                        waterlevels,
                        tri_indices.copy(),
                        tri_vertices,
                        subnb_au,
                        Matrix,
                    )
                    result[index_pts] = interpolated
        result = (np.array([result])).reshape(values.shape)
        return result


class Improvement2(MyCalculator):
    # Interpolation via bilinear interpolation on squares and barycentric on triangles, or IDSW interpolation on squares and triangles.

    def compute_nr_neighbours(self, index, indexD, waterlevels, au, Matrix):
        """ 
        Return:
            how many neighbours cell has that can be used for interpolation.
            Matrix modified for barycentric-interpolation.
            water gives waterlevels of cell and its neighbour when interpolation_type == 2.
        Args:
            index: index of cell.
            indexD: index of diagonal cell.
            waterlevels (array): water levels of cells.
            au: wet cross-sectional area between cell and neighbours.
            Matrix: matrix with 1st row x-coords of cell and its neighbours, 2nd row y-coords of cell and its neighbours, 3rd row ones.
        """
        # initialize
        nbs = [1, 1, 1]
        water = [-1, -1]
        i = -1
        k = 0

        for j in range(3):
            i = i + 1
            if (
                (
                    (not (j == 1))
                    and (
                        (index[j].size == 0)
                        or (waterlevels[index[j]] == self.no_data_value)
                    )
                )
                or ((j == 0) and (au[0] < 10 ** (-7)))
                or ((j == 2) and (au[1] < 10 ** (-7)))
            ):
                # no neighbour-cell with known waterlevels or waterflow
                Matrix = np.delete(Matrix, i, 1)
                i = i - 1
                nbs[j] = 0
            elif k < 2:
                water[k] = waterlevels[index[j]]
                k = k + 1
        if np.sum(nbs) - 1 == 2 and indexD.size > 0:
            return 3, 0, 0

        return np.sum(nbs) - 1, Matrix, water

    def X_diagonal_neighbour(
        self,
        X,
        cell_coords,
        cell_centre,
        coords,
        centres,
        waterlevels,
        subnb_indices,
        lines_wrt_cells,
    ):
        """ 
        Return: index, coordinates of left-bottom and right-top corner and of centre, and type of neighbour to the X position wrt original cell.
        Args:
            X: position of diagonal neighbour, 'lb' = left bottom, 'rb' = right bottom, 'rt' = right top, 'lt' = left top.
            cell_coords: coordinates of cell.
            coords (array): coordinates of left-bottom corner and right-top corner of cells.
            centres (array): coordinates of centres of cells.
            waterlevels (array): water levels of cells.
            subnb_indices: indices of neighbours of subcell.
            lines_wrt_cells (array): cells that lines connects.
        """
        if X == 'lb':
            indexX = np.where(
                (cell_coords[0] > coords[0, :])
                & (cell_coords[1] > coords[1, :])
                & (cell_coords[0] == coords[2, :])
                & (cell_coords[1] == coords[3, :])
            )
        elif X == 'rb':
            indexX = np.where(
                (cell_coords[2] == coords[0, :])
                & (cell_coords[1] > coords[1, :])
                & (cell_coords[2] < coords[2, :])
                & (cell_coords[1] == coords[3, :])
            )
        elif X == 'rt':
            indexX = np.where(
                (cell_coords[2] == coords[0, :])
                & (cell_coords[3] == coords[1, :])
                & (cell_coords[2] < coords[2, :])
                & (cell_coords[3] < coords[3, :])
            )
        else:
            indexX = np.where(
                (cell_coords[0] > coords[0, :])
                & (cell_coords[3] == coords[1, :])
                & (cell_coords[0] == coords[2, :])
                & (cell_coords[3] < coords[3, :])
            )

        line_cell_to_X = [-1, -1]
        sub_nbs_ids = [-1, -1]
        if (indexX[0].size == 0) or (
            waterlevels[indexX] == self.no_data_value
        ):  # no neighbour with known water level
            indexX = np.array([[]])
            centreX = [self.no_data_value, self.no_data_value]
            coordsX = [
                self.no_data_value,
                self.no_data_value,
                self.no_data_value,
                self.no_data_value,
            ]
            typeX = 3
        else:
            cell_to_X_id = indexX[0][0] + 1
            sub_nbs_ids[0] = subnb_indices[0] + 1
            sub_nbs_ids[1] = subnb_indices[1] + 1
            line_cell_to_X[0] = np.where(
                (
                    (lines_wrt_cells[0, :] == cell_to_X_id)
                    & (lines_wrt_cells[1, :] == sub_nbs_ids[0])
                )
                | (
                    (lines_wrt_cells[0, :] == sub_nbs_ids[0])
                    & (lines_wrt_cells[1, :] == cell_to_X_id)
                )
            )
            line_cell_to_X[1] = np.where(
                (
                    (lines_wrt_cells[0, :] == cell_to_X_id)
                    & (lines_wrt_cells[1, :] == sub_nbs_ids[1])
                )
                | (
                    (lines_wrt_cells[0, :] == sub_nbs_ids[1])
                    & (lines_wrt_cells[1, :] == cell_to_X_id)
                )
            )
            if (
                line_cell_to_X[0][0].size < 1 or line_cell_to_X[1][0].size < 1
            ):  # no line between orthogonal and diagonal neighbour
                indexX = np.array([[]])
                centreX = [self.no_data_value, self.no_data_value]
                coordsX = [
                    self.no_data_value,
                    self.no_data_value,
                    self.no_data_value,
                    self.no_data_value,
                ]
                typeX = 3
            else:
                if (
                    cell_coords[2] - cell_coords[0]
                    == coords[2, indexX] - coords[0, indexX]
                ):  # same size
                    centreX = centres[indexX, :][0][0]
                    coordsX = coords[:, indexX[0][0]]
                    typeX = 0
                elif (
                    1 / 2 * (cell_coords[2] - cell_coords[0])
                    == coords[2, indexX] - coords[0, indexX]
                ):  # 1x smaller
                    centreX = centres[indexX, :][0][0]
                    coordsX = coords[:, indexX[0][0]]
                    typeX = 1
                elif (
                    2 * (cell_coords[2] - cell_coords[0])
                    == coords[2, indexX] - coords[0, indexX]
                ):  # 1x larger
                    centreX = centres[indexX, :][0][0]
                    coordsX = coords[:, indexX[0][0]]
                    typeX = 2
                elif (
                    1 / 4 * (cell_coords[2] - cell_coords[0])
                    == coords[2, indexX] - coords[0, indexX]
                ):  # 2x smaller, coarsen diagonal neighbour
                    centreX = centres[indexX, :][0][0]
                    coordsX = coords[:, indexX[0][0]]
                    if X == 'lb':
                        centreX = [coordsX[0], coordsX[1]]
                    elif X == 'rb':
                        centreX = [coordsX[2], coordsX[1]]
                    elif X == 'rt':
                        centreX = [coordsX[2], coordsX[3]]
                    else:
                        centreX = [coordsX[0], coordsX[3]]
                    typeX = 1
                elif (
                    4 * (cell_coords[2] - cell_coords[0])
                    == coords[2, indexX] - coords[0, indexX]
                ):  # 2x larger, refine diagonal neighbour
                    centreX = centres[indexX, :][0][0]
                    coordsX = coords[:, indexX[0][0]]
                    [fine_coordsX, fine_CentresX] = self.refine_cell(coordsX, centreX)
                    centreX = self.find_closest(fine_CentresX, cell_centre)
                    coordsX = self.find_containing_cell(fine_coordsX, centreX)
                    typeX = 2
        return indexX, centreX, coordsX, typeX

    def orth_nb_as_diag_nb(
        self, nb_type, sub_nb_index, nb_index, nb_au, lines_wrt_cells, AU
    ):
        """
        Return index of diagonal neighbour of subcell.
        """
        if nb_type == 0 or nb_type == 2:  # same size/larger orthogonal neighbours
            return sub_nb_index
        else:
            if nb_index.size > 1:  # 2 smaller orhtogonal neighbours
                other_nb = np.where(nb_index != sub_nb_index)
                indexX = nb_index[other_nb]
                if np.all(nb_au > 0):
                    line = np.where(
                        (
                            (lines_wrt_cells[0, :] == nb_index[0] + 1)
                            & (lines_wrt_cells[1, :] == nb_index[1] + 1)
                        )
                        | (
                            (lines_wrt_cells[0, :] == nb_index[1] + 1)
                            & (lines_wrt_cells[1, :] == nb_index[0] + 1)
                        )
                    )
                    au = AU[line]
                    if au > 0:
                        return indexX
        return np.array([[]])

    def find_diag_neighbours_subcell(
        self,
        index,
        subcell,
        sub_coords,
        sub_centre,
        nb_indices,
        ori_nb_au,
        subnb_indices,
        subnb_types,
        dnb_index,
        lines_wrt_cells,
        AU,
    ):
        """ 
        Return indices and coordinates of centres of four diagonal neighbours of subcell.
        """
        sub_dnb_indices = [
            np.array([[]]),
            np.array([[]]),
            np.array([[]]),
            np.array([[]]),
        ]
        sub_dnb_centres = [
            [self.no_data_value, self.no_data_value],
            [self.no_data_value, self.no_data_value],
            [self.no_data_value, self.no_data_value],
            [self.no_data_value, self.no_data_value],
        ]
        subnb_indices = np.array(subnb_indices, dtype=np.ndarray)
        dx = 2 * (sub_centre[0] - sub_coords[0])
        dy = 2 * (sub_centre[1] - sub_coords[1])
        sub_dnb_centres = [
            [sub_centre[0] - dx, sub_centre[1] - dy],
            [sub_centre[0] + dx, sub_centre[1] - dy],
            [sub_centre[0] + dx, sub_centre[1] + dy],
            [sub_centre[0] - dx, sub_centre[1] + dy],
        ]

        if subcell == 1:
            sub_dnb_indices[0] = dnb_index
            sub_dnb_indices[1] = self.orth_nb_as_diag_nb(
                subnb_types[1],
                subnb_indices[1],
                nb_indices[1],
                ori_nb_au[1],
                lines_wrt_cells,
                AU,
            )
            sub_dnb_indices[2] = index
            sub_dnb_indices[3] = self.orth_nb_as_diag_nb(
                subnb_types[0],
                subnb_indices[0],
                nb_indices[0],
                ori_nb_au[0],
                lines_wrt_cells,
                AU,
            )
        elif subcell == 2:
            sub_dnb_indices[0] = self.orth_nb_as_diag_nb(
                subnb_types[0],
                subnb_indices[0],
                nb_indices[1],
                ori_nb_au[1],
                lines_wrt_cells,
                AU,
            )
            sub_dnb_indices[1] = dnb_index
            sub_dnb_indices[2] = self.orth_nb_as_diag_nb(
                subnb_types[1],
                subnb_indices[1],
                nb_indices[2],
                ori_nb_au[2],
                lines_wrt_cells,
                AU,
            )
            sub_dnb_indices[3] = index
        elif subcell == 3:
            sub_dnb_indices[0] = index
            sub_dnb_indices[1] = self.orth_nb_as_diag_nb(
                subnb_types[0],
                subnb_indices[0],
                nb_indices[2],
                ori_nb_au[2],
                lines_wrt_cells,
                AU,
            )
            sub_dnb_indices[2] = dnb_index
            sub_dnb_indices[3] = self.orth_nb_as_diag_nb(
                subnb_types[1],
                subnb_indices[1],
                nb_indices[3],
                ori_nb_au[3],
                lines_wrt_cells,
                AU,
            )
        elif subcell == 4:
            sub_dnb_indices[0] = self.orth_nb_as_diag_nb(
                subnb_types[1],
                subnb_indices[1],
                nb_indices[0],
                ori_nb_au[0],
                lines_wrt_cells,
                AU,
            )
            sub_dnb_indices[1] = index
            sub_dnb_indices[2] = self.orth_nb_as_diag_nb(
                subnb_types[0],
                subnb_indices[0],
                nb_indices[3],
                ori_nb_au[3],
                lines_wrt_cells,
                AU,
            )
            sub_dnb_indices[3] = dnb_index
        return sub_dnb_indices, sub_dnb_centres

    def check_diag_water(self, indexX, typeX, nb_indices, lines_wrt_cells, AU, nb_au):
        """ 
        Return index and type of diagonal neighbour. If the wet cross-sectional area between any of the orthogonal neighbours and cell or diagonal neighbour is zero, 
            then index is empty array and type = 3.
        Args:
            indexX: index of diagonal neighbour.
            typeX: type of diagonal neighbour.
            nb_indices: indices of two orthogonal neighbours.
            lines_wrt_cells (array): cells that lines connects.
            AU (array): wet cross-sectional area at lines.
            nb_au: wet cross-sectional area between cell and its two orthogonal neighbours.
        """
        if (
            indexX[0].size > 0
            and nb_indices[0].size > 0
            and nb_au[0] > 10 ** (-7)
            and nb_indices[1].size > 0
            and nb_au[1] > 10 ** (-7)
        ):
            line1 = np.where(
                (
                    (lines_wrt_cells[0, :] == nb_indices[0] + 1)
                    & (lines_wrt_cells[1, :] == indexX[0] + 1)
                )
                | (
                    (lines_wrt_cells[0, :] == indexX[0] + 1)
                    & (lines_wrt_cells[1, :] == nb_indices[0] + 1)
                )
            )
            line2 = np.where(
                (
                    (lines_wrt_cells[0, :] == nb_indices[1] + 1)
                    & (lines_wrt_cells[1, :] == indexX[0] + 1)
                )
                | (
                    (lines_wrt_cells[0, :] == indexX[0] + 1)
                    & (lines_wrt_cells[1, :] == nb_indices[1] + 1)
                )
            )
            au1 = AU[line1]
            au2 = AU[line2]
            if au1 > 10 ** (-7) and au2 > 10 ** (-7):
                return indexX[0][0], typeX
            else:
                return np.array([[]]), 3
        else:
            return np.array([[]]), 3

    def determine_interpolation_type(
        self, tri_indices, indexD, waterlevels, au, Matrix
    ):
        """ 
        Return:
            interpolation_type: type to be used based on the number and type of neighbours
                (0 = error, cell has no known waterlevel, should not interpolate,
                 1 = no neighbours, use waterlevel of cell
                 2 = 1 neighbour, use average waterlevel based on distance in x- or y-direction
                 3 = 2 neighbours, use barycentric or IDSW interpolation on triangle
                 4 = 3 neighbours, use bilinear or IDSW interpolation on square).
            water: waterlevels of cell and its neighbour when interpolation_type == 2.
            modified_Matrix: Matrix modified for barycentric-interpolation when interpolation_type == 3.
        """
        if waterlevels[tri_indices[1]] == self.no_data_value:
            interpolation_type = 0
            print("error: interpolation type 0")
        nr_nbs, modified_Matrix, water = self.compute_nr_neighbours(
            tri_indices, indexD, waterlevels, au, Matrix
        )
        if nr_nbs == 0:
            interpolation_type = 1
        elif nr_nbs == 1:
            interpolation_type = 2
        elif nr_nbs == 2:
            interpolation_type = 3
        elif nr_nbs == 3:
            interpolation_type = 4
        return interpolation_type, water, modified_Matrix

    def Central_Interpolator(
        self,
        pts_in_sub,
        waterlevels,
        tri_indices,
        tri_vertices,
        au,
        Matrix,
        square_vertices,
        square_indices,
    ):
        """ 
        Return interpolated: array of interpolated waterlevels at points in subcell, based on bilinear or IDSW interpolation on squares when possible,
            otherwise linear barycentric or IDSW interpolation on triangles.
        Args:
            pts_in_sub: array of points to be interpolated in subcell.
            waterlevels: array of waterlevels (s1).
            tri_indices: indices of cells whose centres are vertices of triangle used for interpolation.
            tri_vertices: vertices of triangle used for interpolation.
            au: array of wet cross-sectional area of two neighbour-cells.
            Matrix: matrix with 1st row x-coords, 2nd row y-coords, 3rd row ones.
        """
        nr_points = len(pts_in_sub)
        interpolation_type, water, modified_Matrix = self.determine_interpolation_type(
            tri_indices, square_indices[3], waterlevels, au, Matrix
        )

        if interpolation_type == 1:
            interpolated = waterlevels[tri_indices[1]] * np.ones(nr_points)
            return interpolated
        elif interpolation_type == 2:
            if not (
                modified_Matrix[0][0] - modified_Matrix[0][1] == 0
            ):  # neighbour is to left or right (different x-, same y-coord.s)
                interpolated = self.average_x(pts_in_sub, modified_Matrix, water)
                return interpolated
            elif not (
                modified_Matrix[1][0] - modified_Matrix[1][1] == 0
            ):  # neighbour is to bottom or top (different y-, same x-coord.s)
                interpolated = self.average_y(pts_in_sub, modified_Matrix, water)
                return interpolated
        elif interpolation_type == 3:
            if INTERPOLATION_METHOD == 'barycentric':
                interpolated = self.bary_lin_interpolator(
                    pts_in_sub, tri_indices, modified_Matrix, waterlevels
                )
            elif INTERPOLATION_METHOD == 'distance':
                interpolated = self.triangle_IDSW_interpolator(
                    pts_in_sub, waterlevels, tri_indices, tri_vertices
                )
            elif INTERPOLATION_METHOD == 'combi':
                interpolated = self.triangle_IDSW_interpolator(
                    pts_in_sub, waterlevels, tri_indices, tri_vertices
                )
            return interpolated
        elif interpolation_type == 4:
            if INTERPOLATION_METHOD == 'barycentric':
                interpolated = self.bilinear_interpolator(
                    pts_in_sub, square_vertices, square_indices, waterlevels
                )
            elif INTERPOLATION_METHOD == 'distance':
                interpolated = self.square_IDSW_interpolator(
                    pts_in_sub, square_vertices, square_indices, waterlevels
                )
            elif INTERPOLATION_METHOD == 'combi':
                interpolated = self.bilinear_interpolator(
                    pts_in_sub, square_vertices, square_indices, waterlevels
                )
            return interpolated

    def refine_neighbours(self, subnb_types, subnb_coords, subnb_centres):
        """ 
        Return type, coordinates and centres of refined cells of original cell that are also neighbours of the subcell.
        Args:
            subnb_types: type of neighbours of subcell.
            subnb_coords: coordinates of neighbours of subcell.
            subnb_centres: coordinates of centres of neighbours of subcell.
        """
        if subnb_types[0] == 1 or subnb_types[0] == 3:  # smaller or nonexistent
            centre_nb1 = subnb_centres[0]
        if subnb_types[1] == 1 or subnb_types[1] == 3:  # smaller or nonexistent
            centre_nb2 = subnb_centres[1]
        if subnb_types[0] == 0:  # same size, refine once
            nb1_fine_coords, nb1_fine_centres = self.refine_cell(
                subnb_coords[0], subnb_centres[0]
            )
            centre_nb1 = self.find_closest(nb1_fine_centres, subnb_centres[1])
        if subnb_types[1] == 0:  # same size, refine once
            nb2_fine_coords, nb2_fine_centres = self.refine_cell(
                subnb_coords[1], subnb_centres[1]
            )
            centre_nb2 = self.find_closest(nb2_fine_centres, centre_nb1)
        if subnb_types[0] == 2:  # larger, refine twice
            nb1_fine_coords1, nb1_fine_centres1 = self.refine_cell(
                subnb_coords[0], subnb_centres[0]
            )
            centre_nb11 = self.find_closest(nb1_fine_centres1, centre_nb2)
            coords_nb11 = self.find_containing_cell(nb1_fine_coords1, centre_nb11)
            nb1_fine_coords, nb1_fine_centres = self.refine_cell(
                coords_nb11, centre_nb11
            )
            centre_nb1 = self.find_closest(nb1_fine_centres, centre_nb2)
        if subnb_types[1] == 2:  # larger, refine twice
            nb2_fine_coords1, nb2_fine_centres1 = self.refine_cell(
                subnb_coords[1], subnb_centres[1]
            )
            centre_nb21 = self.find_closest(nb2_fine_centres1, centre_nb1)
            coords_nb21 = self.find_containing_cell(nb2_fine_coords1, centre_nb21)
            nb2_fine_coords, nb2_fine_centres = self.refine_cell(
                coords_nb21, centre_nb21
            )
            centre_nb2 = self.find_closest(nb2_fine_centres, centre_nb1)
        return centre_nb1, centre_nb2

    def find_diag_neighbour_centre(self, dnb_type, dnb_centre, dnb_coords, sub_centre):
        """ 
        Return centre of diagonal neighbour when one of other neighbours is smaller.
        """
        if dnb_type == 1 or dnb_type == 3:  # smaller or nonexistent
            centre_dnb = dnb_centre
        if dnb_type == 0:  # same size, refine once
            dnb_fine_coords, dnb_fine_centres = self.refine_cell(dnb_coords, dnb_centre)
            centre_dnb = self.find_closest(dnb_fine_centres, sub_centre)
        if dnb_type == 2:  # larger, refine twice
            dnb_fine_coords1, dnb_fine_centres1 = self.refine_cell(
                dnb_coords, dnb_centre
            )
            centre_dnb1 = self.find_closest(dnb_fine_centres1, sub_centre)
            coords_dnb1 = self.find_containing_cell(dnb_fine_coords1, centre_dnb1)
            dnb_fine_coords, dnb_fine_centres = self.refine_cell(
                coords_dnb1, centre_dnb1
            )
            centre_dnb = self.find_closest(dnb_fine_centres, sub_centre)
        return centre_dnb

    def small_neighbour_prep(
        self,
        index,
        cell_coords,
        cell_centre,
        nb_indices,
        nb_au,
        subcell,
        sub_coords,
        sub_centre,
        subnb_indices,
        subnb_coords,
        subnb_centres,
        subnb_au,
        subnb_types,
        dnb_index,
        dnb_type,
        dnb_centre,
        dnb_coords,
        lines_wrt_cells,
        AU,
    ):
        """ 
        Return:
            sub_fine_coords: coordinates of left-bottom and right-top corners of refined cells.
            subsubnbs_indices: indices of 4 orthogonal neighbours of subcell.
            subsubnbs_centres: coordinates of centres of 4 orthogonal neighbours of subcell.
            subsubnbs_au: wet-cross secitonal area of 4 orthogonal neighbours of subcell.
            subsubdnbs_indices: indices of 4 diagonal neighbours of subcell.
            subsubdnbs_centres: coordinates of centres of 4 diagonal neighbours of subcell.
        """
        # One of neighbours is small. Cell is refined and if necessary neighbours are refined. Subcell is refined.
        fine_coords, fine_centres = self.refine_cell(cell_coords, cell_centre)
        centre_nb1, centre_nb2 = self.refine_neighbours(
            subnb_types, subnb_coords, subnb_centres
        )
        sub_fine_coords, sub_fine_centres = self.refine_cell(sub_coords, sub_centre)
        centre_dnb = self.find_diag_neighbour_centre(
            dnb_type, dnb_centre, dnb_coords, sub_centre
        )
        subsubdnbs_indices, subsubdnbs_centres = self.find_diag_neighbours_subcell(
            index,
            subcell,
            sub_coords,
            sub_centre,
            nb_indices,
            nb_au,
            subnb_indices,
            subnb_types,
            dnb_index,
            lines_wrt_cells,
            AU,
        )
        subnb_indices = np.array(subnb_indices, dtype=np.ndarray)
        if subcell == 1:
            subsubnbs_indices = [subnb_indices[0], subnb_indices[1], index, index]
            subsubnbs_centres = [
                centre_nb1,
                centre_nb2,
                fine_centres[1],
                fine_centres[3],
            ]
            subsubnbs_au = [subnb_au[0], subnb_au[1], 1, 1]
        elif subcell == 2:
            subsubnbs_indices = [index, subnb_indices[0], subnb_indices[1], index]
            subsubnbs_centres = [
                fine_centres[0],
                centre_nb1,
                centre_nb2,
                fine_centres[2],
            ]
            subsubnbs_au = [1, subnb_au[0], subnb_au[1], 1]
        elif subcell == 3:
            subsubnbs_indices = [index, index, subnb_indices[0], subnb_indices[1]]
            subsubnbs_centres = [
                fine_centres[3],
                fine_centres[1],
                centre_nb1,
                centre_nb2,
            ]
            subsubnbs_au = [1, 1, subnb_au[0], subnb_au[1]]
        else:
            subsubnbs_indices = [subnb_indices[1], index, index, subnb_indices[0]]
            subsubnbs_centres = [
                centre_nb2,
                fine_centres[0],
                fine_centres[2],
                centre_nb1,
            ]
            subsubnbs_au = [subnb_au[1], 1, 1, subnb_au[0]]
        return (
            sub_fine_coords,
            subsubnbs_indices,
            subsubnbs_centres,
            subsubnbs_au,
            subsubdnbs_indices,
            subsubdnbs_centres,
        )

    def small_neighbour_interpolator(
        self,
        index,
        sub_centre,
        waterlevels,
        pts_in_sub,
        index_pts_in_sub,
        sub_fine_coords,
        subsubnbs_indices,
        subsubnbs_centres,
        subsubnbs_au,
        subsubdnbs_indices,
        subsubdnbs_centres,
        result,
    ):
        """ 
        Return result, containing interpolated values at indices of points in subcell.
        """
        for subsubcell in range(1, 5):
            subsub_coords = sub_fine_coords[subsubcell - 1]
            subtri_vertices, subtri_indices, subau = self.subsubcell_info(
                index,
                sub_centre,
                subsubnbs_indices,
                subsubnbs_centres,
                subsubnbs_au,
                subsubcell,
            )
            subsquare_vertices = subtri_vertices.copy()
            subsquare_vertices.append(subsubdnbs_centres[subsubcell - 1])
            subsquare_indices = subtri_indices.copy()
            subsquare_indices.append(subsubdnbs_indices[subsubcell - 1])
            pts_in_subsub, subindex_pts = self.find_points_subcell(
                pts_in_sub, subsub_coords, index_pts_in_sub
            )
            if pts_in_subsub.size == 0:
                continue
            subMatrix = [
                [subtri_vertices[0][0], subtri_vertices[1][0], subtri_vertices[2][0]],
                [subtri_vertices[0][1], subtri_vertices[1][1], subtri_vertices[2][1]],
                [1, 1, 1],
            ]
            subinterpolated = self.Central_Interpolator(
                pts_in_subsub,
                waterlevels,
                subtri_indices,
                subtri_vertices,
                subau,
                subMatrix,
                subsquare_vertices,
                subsquare_indices,
            )
            result[subindex_pts] = subinterpolated
        return result

    def orthog_is_diag(
        self,
        subcell,
        subnb_types,
        subnb_indices,
        subnb_centres,
        subnb_coords,
        cell_centre,
        diag_nb_centre,
        diag_nb_index,
    ):
        """ 
        Return centre and index of diagonal neighbour when larger orthogonal neighbour can also be used as diagonal neighbour.
        """
        if subcell == 1:
            if subnb_types[0] == 2 and subnb_centres[0][1] < cell_centre[1]:
                nb1_fine_coords, nb1_fine_centres = self.refine_cell(
                    subnb_coords[0], subnb_centres[0]
                )
                centre_nbd = nb1_fine_centres[1]
                diag_nb_index = subnb_indices[0]
            elif subnb_types[1] == 2 and subnb_centres[1][0] < cell_centre[0]:
                nb2_fine_coords, nb2_fine_centres = self.refine_cell(
                    subnb_coords[1], subnb_centres[1]
                )
                centre_nbd = nb2_fine_centres[3]
                diag_nb_index = subnb_indices[1]
            else:
                centre_nbd = diag_nb_centre
        elif subcell == 2:
            if subnb_types[0] == 2 and subnb_centres[0][0] > cell_centre[0]:
                nb1_fine_coords, nb1_fine_centres = self.refine_cell(
                    subnb_coords[0], subnb_centres[0]
                )
                centre_nbd = nb1_fine_centres[2]
                diag_nb_index = subnb_indices[0]
            elif subnb_types[1] == 2 and subnb_centres[1][1] < cell_centre[1]:
                nb2_fine_coords, nb2_fine_centres = self.refine_cell(
                    subnb_coords[1], subnb_centres[1]
                )
                centre_nbd = nb2_fine_centres[0]
                diag_nb_index = subnb_indices[1]
            else:
                centre_nbd = diag_nb_centre
        elif subcell == 3:
            if subnb_types[0] == 2 and subnb_centres[0][1] > cell_centre[1]:
                nb1_fine_coords, nb1_fine_centres = self.refine_cell(
                    subnb_coords[0], subnb_centres[0]
                )
                centre_nbd = nb1_fine_centres[3]
                diag_nb_index = subnb_indices[0]
            elif subnb_types[1] == 2 and subnb_centres[1][0] > cell_centre[0]:
                nb2_fine_coords, nb2_fine_centres = self.refine_cell(
                    subnb_coords[1], subnb_centres[1]
                )
                centre_nbd = nb2_fine_centres[1]
                diag_nb_index = subnb_indices[1]
            else:
                centre_nbd = diag_nb_centre
        elif subcell == 4:
            if subnb_types[0] == 2 and subnb_centres[0][0] < cell_centre[0]:
                nb1_fine_coords, nb1_fine_centres = self.refine_cell(
                    subnb_coords[0], subnb_centres[0]
                )
                centre_nbd = nb1_fine_centres[0]
                diag_nb_index = subnb_indices[0]
            elif subnb_types[1] == 2 and subnb_centres[1][1] > cell_centre[1]:
                nb2_fine_coords, nb2_fine_centres = self.refine_cell(
                    subnb_coords[1], subnb_centres[1]
                )
                centre_nbd = nb2_fine_centres[2]
                diag_nb_index = subnb_indices[1]
            else:
                centre_nbd = diag_nb_centre
        return centre_nbd, diag_nb_index

    def large_neighbour_interpolator(
        self,
        index,
        subcell,
        cell_centre,
        pts_in_sub,
        subnb_types,
        subnb_indices,
        subnb_coords,
        subnb_centres,
        diag_nb_index,
        diag_nb_type,
        diag_nb_coords,
        diag_nb_centre,
        waterlevels,
        au,
    ):
        """ 
        Return interpolated: array of interpolated waterlevels at points in subcell.
        """
        # One of neighbours is large, no neighbours are small.
        if subnb_types[0] == 2:  # large, refine once
            nb1_fine_coords, nb1_fine_centres = self.refine_cell(
                subnb_coords[0], subnb_centres[0]
            )
            centre_nb1 = self.find_closest(nb1_fine_centres, cell_centre)
        else:  # same size or nonexistent
            centre_nb1 = subnb_centres[0]
        if subnb_types[1] == 2:  # large, refine once
            nb2_fine_coords, nb2_fine_centres = self.refine_cell(
                subnb_coords[1], subnb_centres[1]
            )
            centre_nb2 = self.find_closest(nb2_fine_centres, cell_centre)
        else:  # same size or nonexistent
            centre_nb2 = subnb_centres[1]
        if diag_nb_type == 2:  # large, refine once
            nbd_fine_coords, nbd_fine_centres = self.refine_cell(
                diag_nb_coords, diag_nb_centre
            )
            centre_nbd = self.find_closest(nbd_fine_centres, cell_centre)
        elif diag_nb_type == 3:  # larger orthogonal neighbour can be diagonal neighbour
            centre_nbd, diag_nb_index = self.orthog_is_diag(
                subcell,
                subnb_types,
                subnb_indices,
                subnb_centres,
                subnb_coords,
                cell_centre,
                diag_nb_centre,
                diag_nb_index.copy(),
            )
        else:  # same size or nonexistent
            centre_nbd = diag_nb_centre

        tri_vertices = [centre_nb1, cell_centre, centre_nb2]
        tri_indices = np.array(
            [np.array([subnb_indices[0]]), index, np.array([subnb_indices[1]])],
            dtype=np.ndarray,
        )
        Matrix = [
            [tri_vertices[0][0], tri_vertices[1][0], tri_vertices[2][0]],
            [tri_vertices[0][1], tri_vertices[1][1], tri_vertices[2][1]],
            [1, 1, 1],
        ]
        square_vertices = [
            [tri_vertices[0][0], tri_vertices[0][1]],
            [tri_vertices[1][0], tri_vertices[1][1]],
            [tri_vertices[2][0], tri_vertices[2][1]],
            centre_nbd,
        ]
        square_indices = [tri_indices[0], tri_indices[1], tri_indices[2], diag_nb_index]
        interpolated = self.Central_Interpolator(
            pts_in_sub,
            waterlevels,
            tri_indices,
            tri_vertices,
            au,
            Matrix,
            square_vertices,
            square_indices,
        )
        return interpolated

    def bilinear_interpolator(
        self, pts_in_sub, square_vertices, square_indices, waterlevels
    ):
        """ 
        Return interpolated (array): bilinearly interpolated water levels at points in subcell.
        """
        C = np.array(
            [
                waterlevels[square_indices[0]],
                waterlevels[square_indices[1]],
                waterlevels[square_indices[2]],
                waterlevels[square_indices[3]],
            ],
            dtype=float,
        )
        biMatrix = np.matrix(
            [
                [
                    1,
                    square_vertices[0][0],
                    square_vertices[0][1],
                    square_vertices[0][0] * square_vertices[0][1],
                ],
                [
                    1,
                    square_vertices[1][0],
                    square_vertices[1][1],
                    square_vertices[1][0] * square_vertices[1][1],
                ],
                [
                    1,
                    square_vertices[2][0],
                    square_vertices[2][1],
                    square_vertices[2][0] * square_vertices[2][1],
                ],
                [
                    1,
                    square_vertices[3][0],
                    square_vertices[3][1],
                    square_vertices[3][0] * square_vertices[3][1],
                ],
            ],
            dtype='float',
        )
        bi_arr = np.c_[
            np.ones(len(pts_in_sub)),
            pts_in_sub[:, 0],
            pts_in_sub[:, 1],
            pts_in_sub[:, 0] * pts_in_sub[:, 1],
        ].transpose()
        b = np.dot((np.linalg.inv(biMatrix)).transpose(), bi_arr)
        interpolated = b.transpose() @ C
        return interpolated

    def square_IDSW_interpolator(
        self, pts_in_sub, square_vertices, square_indices, waterlevels
    ):
        """ 
        Return interpolated (array): IDSW (on squares) interpolated water levels.
        """

        def distance(pts, pt):  # computes distance between array of points and a point
            return np.sqrt((pts[:, 0] - pt[0]) ** 2 + (pts[:, 1] - pt[1]) ** 2)

        p = 2
        C = np.array(
            [
                waterlevels[square_indices[0]],
                waterlevels[square_indices[1]],
                waterlevels[square_indices[2]],
                waterlevels[square_indices[3]],
            ],
            dtype=float,
        )
        dist_pt_1 = distance(pts_in_sub, square_vertices[0][:])
        dist_pt_2 = distance(pts_in_sub, square_vertices[1][:])
        dist_pt_3 = distance(pts_in_sub, square_vertices[2][:])
        dist_pt_4 = distance(pts_in_sub, square_vertices[3][:])
        frac1 = np.zeros(len(pts_in_sub))
        frac2 = np.zeros(len(pts_in_sub))
        frac3 = np.zeros(len(pts_in_sub))
        frac4 = np.zeros(len(pts_in_sub))
        ind = []
        # avoid division by 0:
        if np.any(dist_pt_1 == 0):
            ind1 = np.where(dist_pt_1 == 0)
            ind = ind + ind1
            frac1[ind1] = 1
        if np.any(dist_pt_2 == 0):
            ind2 = np.where(dist_pt_2 == 0)
            ind = ind + ind2
            frac2[ind2] = 1
        if np.any(dist_pt_3 == 0):
            ind3 = np.where(dist_pt_3 == 0)
            ind = ind + ind3
            frac3[ind3] = 1
        if np.any(dist_pt_4 == 0):
            ind4 = np.where(dist_pt_4 == 0)
            ind = ind + ind4
            fract[ind4] = 1
        index = np.ones(len(pts_in_sub), bool)
        index[ind] = False

        frac1[index] = 1 / dist_pt_1[index] ** p
        frac2[index] = 1 / dist_pt_2[index] ** p
        frac3[index] = 1 / dist_pt_3[index] ** p
        frac4[index] = 1 / dist_pt_4[index] ** p
        total_frac = frac1 + frac2 + frac3 + frac4
        weight1 = frac1 / total_frac
        weight2 = frac2 / total_frac
        weight3 = frac3 / total_frac
        weight4 = frac4 / total_frac
        interpolated = weight1 * C[0] + weight2 * C[1] + weight3 * C[2] + weight4 * C[3]
        return interpolated

    def __call__(self, indices, values, no_data_value):
        """ 
        Return result containing interpolated waterlevels. 
            Interpolation done by combination of bilinear and linear barycentric interpolation if INTERPOLATION_METHOD == 'barycentric'
            Interpolation done by combination of IDSW on squares and triangles if INTERPOLATION_METHOD == 'distance'
        Args:
            indices: indicies = (yoff, xoff), (yoff + ysize, xoff + xsize) in partition.
            values: result needs to have same shape.
            no_data_value: -9999
        """
        print(indices)

        # interpolating points
        points = self._get_points(indices)
        nr_points = len(points)
        pt_to_cell = self._get_nodgrid(indices).reshape(
            nr_points
        )  # array giving for each point in which cell it is.

        # initialize result
        self.no_data_value = no_data_value
        result = np.full(nr_points, self.no_data_value)

        max_nodes, max_lines = self.nr_nodes_lines
        # info nodes/cells
        x = np.arange(1, max_nodes + 1).reshape(1, -1)
        nodes_id_s1_coords = self.lookup_nodes[x][0]
        centres = nodes_id_s1_coords[:, 1:3]  # cell-centre coordinates
        waterlevels = nodes_id_s1_coords[:, 0]  # waterlevels in 2D-open-water cells
        coords = nodes_id_s1_coords[
            :, 3:7
        ].transpose()  # coords of cell left-bottom corner and right-top corner

        # info lines
        x_lines = np.arange(0, max_lines).reshape(1, -1)
        lines_line_linecoords_au = self.lookup_lines[x_lines][0]
        lines_wrt_cells = lines_line_linecoords_au[
            :, 0:2
        ].transpose()  # 2D array of cell-ids that the line connects
        AU = lines_line_linecoords_au[:, 6].transpose()  # wet cross-sectional area

        # find in which cells interpolating points are contained
        indexrange = np.unique(pt_to_cell)
        indexrange = indexrange[indexrange != 0]

        # loop over cells that contain interpolating points
        for cell_id in indexrange:
            # info cell
            index = cell_id - 1
            cell_coords = coords[:, index]
            cell_centre_true = [centres[index, 0], centres[index, 1]]
            cell_centre = cell_centre_true.copy()

            # info points in cell
            index_pts_in_cell = np.where(pt_to_cell == cell_id)[0]
            pts_in_cell = points[index_pts_in_cell]

            if (
                waterlevels[index] == no_data_value
            ):  # check if is waterlevel available for cell
                result[index_pts_in_cell] = no_data_value
                continue

            kwargs_neighbour = dict(
                cell_coords=cell_coords, cell_centre=cell_centre, coords=coords
            )
            kwargs_info = dict(
                coords=coords,
                centres=centres,
                lines_wrt_cells=lines_wrt_cells,
                AU=AU,
                cell_id=cell_id,
            )
            # cell to left
            index_left, type_left = self.X_neighbour('l', **kwargs_neighbour)
            coords_left, centre_left, au_left = self.cell_info(
                index_left, **kwargs_info
            )

            # cell to right
            index_right, type_right = self.X_neighbour('r', **kwargs_neighbour)
            coords_right, centre_right, au_right = self.cell_info(
                index_right, **kwargs_info
            )

            # cell to bottom
            index_bottom, type_bottom = self.X_neighbour('b', **kwargs_neighbour)
            coords_bottom, centre_bottom, au_bottom = self.cell_info(
                index_bottom, **kwargs_info
            )

            # cell to top
            index_top, type_top = self.X_neighbour('t', **kwargs_neighbour)
            coords_top, cell_to_top_centre, au_top = self.cell_info(
                index_top, **kwargs_info
            )

            nb_indices = [index_left, index_bottom, index_right, index_top]
            nb_coords = [coords_left, coords_bottom, coords_right, coords_top]
            nb_centres = [
                centre_left.copy(),
                centre_bottom.copy(),
                centre_right.copy(),
                cell_to_top_centre.copy(),
            ]
            nb_au = [au_left, au_bottom, au_right, au_top]
            nb_type = np.array([type_left, type_bottom, type_right, type_top])

            # loop over subcell/quadrant of each cell
            for subcell in range(1, 5):
                cell_centre = cell_centre_true.copy()
                # info subcell
                subnb_indices, subnb_coords, subnb_centres, subnb_au, subnb_types = self.find_neighbours_subcell(
                    subcell,
                    cell_centre,
                    nb_type,
                    nb_indices,
                    nb_coords,
                    nb_centres,
                    nb_au,
                )
                sub_coords, tri_indices, tri_vertices = self.subcell_info(
                    index,
                    cell_coords,
                    cell_centre,
                    subcell,
                    subnb_indices,
                    subnb_centres,
                )
                sub_centre = [
                    sub_coords[0] + (sub_coords[2] - sub_coords[0]) / 2,
                    sub_coords[1] + (sub_coords[3] - sub_coords[1]) / 2,
                ]

                # find interpolating points in each subcel
                pts_in_sub, index_pts = self.find_points_subcell(
                    pts_in_cell, sub_coords, index_pts_in_cell
                )
                if pts_in_sub.size == 0:  # no points in subcell
                    continue

                # info diagonal neighbour
                diag_loc = ['lb', 'rb', 'rt', 'lt']
                dnb_index_pre, dnb_centre, dnb_coords, dnb_type = self.X_diagonal_neighbour(
                    diag_loc[subcell - 1],
                    cell_coords,
                    cell_centre,
                    coords,
                    centres,
                    waterlevels,
                    subnb_indices,
                    lines_wrt_cells,
                )
                dnb_index, dnb_type = self.check_diag_water(
                    dnb_index_pre,
                    dnb_type,
                    subnb_indices,
                    lines_wrt_cells,
                    AU,
                    subnb_au,
                )

                if (
                    np.any(subnb_types == 1) or dnb_type == 1
                ):  # at least 1 neighbour smaller
                    sub_fine_coords, subsubnbs_indices, subsubnbs_centres, subsubnbs_au, subsubdnbs_indices, subsubdnbs_centres = self.small_neighbour_prep(
                        index,
                        cell_coords,
                        cell_centre,
                        nb_indices,
                        nb_au,
                        subcell,
                        sub_coords,
                        sub_centre,
                        subnb_indices,
                        subnb_coords,
                        subnb_centres,
                        subnb_au,
                        subnb_types,
                        dnb_index,
                        dnb_type,
                        dnb_centre,
                        dnb_coords,
                        lines_wrt_cells,
                        AU,
                    )
                    result = self.small_neighbour_interpolator(
                        index,
                        sub_centre,
                        waterlevels,
                        pts_in_sub,
                        index_pts,
                        sub_fine_coords,
                        subsubnbs_indices,
                        subsubnbs_centres,
                        subsubnbs_au,
                        subsubdnbs_indices,
                        subsubdnbs_centres,
                        result,
                    )
                elif (
                    np.any(subnb_types == 2) or dnb_type == 2
                ):  # at least 1 neighbour larger
                    interpolated = self.large_neighbour_interpolator(
                        index,
                        subcell,
                        cell_centre,
                        pts_in_sub,
                        subnb_types,
                        subnb_indices,
                        subnb_coords,
                        subnb_centres,
                        dnb_index,
                        dnb_type,
                        dnb_coords,
                        dnb_centre,
                        waterlevels,
                        subnb_au,
                    )
                    result[index_pts] = interpolated
                else:  # all same size or no neighbours
                    Matrix = [
                        [tri_vertices[0][0], tri_vertices[1][0], tri_vertices[2][0]],
                        [tri_vertices[0][1], tri_vertices[1][1], tri_vertices[2][1]],
                        [1, 1, 1],
                    ]
                    square_vertices = [
                        [tri_vertices[0][0], tri_vertices[0][1]],
                        [tri_vertices[1][0], tri_vertices[1][1]],
                        [tri_vertices[2][0], tri_vertices[2][1]],
                        dnb_centre,
                    ]
                    square_indices = [
                        tri_indices[0],
                        tri_indices[1],
                        tri_indices[2],
                        dnb_index,
                    ]
                    interpolated = self.Central_Interpolator(
                        pts_in_sub,
                        waterlevels,
                        tri_indices.copy(),
                        tri_vertices,
                        subnb_au,
                        Matrix,
                        square_vertices,
                        square_indices,
                    )
                    result[index_pts] = interpolated
        result = (np.array([result])).reshape(values.shape)
        return result


class MyDepthCalculator1(Improvement1):
    def __call__(self, indices, values, no_data_value):
        """Return waterdepth array."""
        waterlevel = super().__call__(indices, values, no_data_value)
        return self._depth_from_water_level(
            dem=values, fillvalue=no_data_value, waterlevel=waterlevel
        )


class MyDepthCalculator2(Improvement2):
    def __call__(self, indices, values, no_data_value):
        """Return waterdepth array."""
        waterlevel = super().__call__(indices, values, no_data_value)
        return self._depth_from_water_level(
            dem=values, fillvalue=no_data_value, waterlevel=waterlevel
        )


class CopyCalculator(Calculator):
    def __call__(self, indices, values, no_data_value):
        """Return input values unmodified."""
        return values


class NodGridCalculator(Calculator):
    def __call__(self, indices, values, no_data_value):
        """Return node grid."""
        return self._get_nodgrid(indices)


class ConstantLevelCalculator(Calculator):
    def __call__(self, indices, values, no_data_value):
        """Return waterlevel array."""
        return self.lookup_s1[self._get_nodgrid(indices)]


class InterpolatedLevelCalculator(Calculator):
    def __call__(self, indices, values, no_data_value):
        """Return waterlevel array."""
        points = self._get_points(indices)
        return self.interpolator(points).reshape(values.shape)


class ConstantLevelDepthCalculator(ConstantLevelCalculator):
    def __call__(self, indices, values, no_data_value):
        """Return waterdepth array."""
        waterlevel = super().__call__(indices, values, no_data_value)
        return self._depth_from_water_level(
            dem=values, fillvalue=no_data_value, waterlevel=waterlevel
        )


class InterpolatedLevelDepthCalculator(InterpolatedLevelCalculator):
    def __call__(self, indices, values, no_data_value):
        """Return waterdepth array."""
        waterlevel = super().__call__(indices, values, no_data_value)
        return self._depth_from_water_level(
            dem=values, fillvalue=no_data_value, waterlevel=waterlevel
        )


class GeoTIFFConverter:
    """Convert tiff, applying a calculating function to the data.
    Args:
        source_path (str): Path to source GeoTIFF file.
        target_path (str): Path to target GeoTIFF file.
        progress_func: a callable.
        The progress_func will be called multiple times with values between 0.0
        amd 1.0.
    """

    def __init__(
        self,
        gridadmin_path,
        results_3di_path,
        calculation_step,
        source_path,
        target_path,
        progress_func=None,
    ):
        self.source_path = source_path
        self.target_path = target_path

        self.progress_func = progress_func

        self.gridadmin_path = gridadmin_path
        self.results_3di_path = results_3di_path
        self.calculation_step = calculation_step

        if path.exists(self.target_path):
            raise OSError("%s already exists." % self.target_path)

    def __enter__(self):
        """Open datasets.
        """
        self.source = gdal.Open(self.source_path, gdal.GA_ReadOnly)
        block_x_size, block_y_size = self.block_size
        options = ["compress=deflate", "blockysize=%s" % block_y_size]
        if block_x_size != self.raster_x_size:
            options += ["tiled=yes", "blockxsize=%s" % block_x_size]

        self.target = gdal.GetDriverByName("gtiff").Create(
            self.target_path,
            self.raster_x_size,
            self.raster_y_size,
            1,  # band count
            self.source.GetRasterBand(1).DataType,
            options=options,
        )
        self.target.SetProjection(self.projection)
        self.target.SetGeoTransform(self.geo_transform)
        self.target.GetRasterBand(1).SetNoDataValue(self.no_data_value)

        return self

    def __exit__(self, *args):
        """Close datasets.
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
        self.gr = GridH5ResultAdmin(self.gridadmin_path, self.results_3di_path)
        nodes2 = self.gr.nodes.subset(SUBSET_2D_OPEN_WATER)
        timeseries2 = nodes2.timeseries(indexes=[self.calculation_step])
        data2 = timeseries2.only("cell_coords").data
        cell_coords2 = data2["cell_coords"]
        STEP = int(np.max(cell_coords2[2, :] - cell_coords2[0, :]))

        def offset_size_range(stop, step):
            for start in range(0, stop, step):
                yield start, min(step, stop - start)

        # tiled tiff writing is much faster row-wise
        raster_size = self.raster_y_size, self.raster_x_size
        block_size = self.block_size[::-1]

        total_size = block_size[0] * block_size[1]
        if block_size[0] <= STEP:
            block_size[0] = STEP
        else:
            block_size[0] = block_size[0] - (block_size[0] % STEP)
        if block_size[1] < STEP:
            block_size[1] = STEP

        offset_size_range
        generator = product(*map(offset_size_range, raster_size, block_size))

        total = len(self)
        for count, result in enumerate(generator, start=1):
            yield result[::-1]
            if self.progress_func is not None:
                self.progress_func(count / total)

    def convert_using(self, calculator):
        """Convert data writing it to tiff. """
        no_data_value = self.no_data_value
        for (xoff, xsize), (yoff, ysize) in self.partition():
            values = self.source.ReadAsArray(
                xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize
            )
            indices = (yoff, xoff), (yoff + ysize, xoff + xsize)
            result = calculator(
                indices=indices, values=values, no_data_value=no_data_value
            )

            self.target.GetRasterBand(1).WriteArray(array=result, xoff=xoff, yoff=yoff)


calculator_classes = {
    MODE_COPY: CopyCalculator,
    MODE_NODGRID: NodGridCalculator,
    MODE_CONSTANT_S1: ConstantLevelCalculator,
    MODE_INTERPOLATED_S1: InterpolatedLevelCalculator,
    MODE_CONSTANT: ConstantLevelDepthCalculator,
    MODE_INTERPOLATED: InterpolatedLevelDepthCalculator,
    MODE_MY_INTERP_1: Improvement1,
    MODE_MY_INTERP_1_DEPTH: MyDepthCalculator1,
    MODE_MY_INTERP_2: Improvement2,
    MODE_MY_INTERP_2_DEPTH: MyDepthCalculator2,
}


def calculate_waterdepth(
    gridadmin_path,
    results_3di_path,
    dem_path,
    waterdepth_path,
    calculation_step=-1,
    mode=MODE_INTERPOLATED,
    progress_func=None,
):
    """Calculate waterdepth and save it as GeoTIFF.
    Args:
        gridadmin_path (str): Path to gridadmin.h5 file.
        results_3di_path (str): Path to results_3di.nc file.
        dem_path (str): Path to dem.tif file.
        waterdepth_path (str): Path to waterdepth.tif file.
        calculation_step (int): Calculation step (default: -1 (last))
        interpolate (bool): Interpolate linearly between nodes.
    """
    try:
        CalculatorClass = calculator_classes[mode]
    except KeyError:
        raise ValueError("Unknown mode: '%s'" % mode)

    # TODO remove at some point, newly produced gridadmins don't need it
    fix_gridadmin(gridadmin_path)

    converter_kwargs = {
        "gridadmin_path": gridadmin_path,
        "results_3di_path": results_3di_path,
        "calculation_step": calculation_step,
        "source_path": dem_path,
        "target_path": waterdepth_path,
        "progress_func": progress_func,
    }

    with GeoTIFFConverter(**converter_kwargs) as converter:

        calculator_kwargs = {
            "gridadmin_path": gridadmin_path,
            "results_3di_path": results_3di_path,
            "calculation_step": calculation_step,
            "dem_geo_transform": converter.geo_transform,
            "dem_pixelsize": converter.geo_transform[1],
            "dem_shape": (converter.raster_y_size, converter.raster_x_size),
        }
        with CalculatorClass(**calculator_kwargs) as calculator:
            converter.convert_using(calculator)
