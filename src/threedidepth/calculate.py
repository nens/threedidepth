# -*- coding: utf-8 -*-

from itertools import product
from os import path

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from osgeo import gdal
from threedigrid.admin.gridresultadmin import GridH5ResultAdmin
from threedigrid.admin.gridresultadmin import GridH5Admin
from threedigrid.admin.constants import SUBSET_2D_OPEN_WATER
from threedigrid.admin.constants import NO_DATA_VALUE
from threedidepth.fixes import fix_gridadmin

MODE_COPY = "copy"
MODE_NODGRID = "nodgrid"
MODE_CONSTANT_S1 = "constant-s1"
MODE_INTERPOLATED_S1 = "interpolated-s1"
MODE_CONSTANT = "constant"
MODE_INTERPOLATED = "interpolated"
MODE_TEST = "test"


class Calculator:
    """Depth calculator using constant waterlevel in a grid cell.

    Args:
        gridadmin_path (str): Path to gridadmin.h5 file.
        results_3di_path (str): Path to results_3di.nc file.
        calculation_step (int): Calculation step for the waterdepth.
        dem_shape (int, int): Shape of the dem array.
        dem_geo_transform: (tuple) Geo_transform of the dem.
    """
    PIXEL_MAP = "pixel_map"
    LOOKUP_S1 = "lookup_s1"
    INTERPOLATOR = "interpolator"
    TESTINTERPOLATOR = "testinterpolator"
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
    ):
        self.gridadmin_path = gridadmin_path
        self.results_3di_path = results_3di_path
        self.calculation_step = calculation_step
        self.dem_shape = dem_shape
        self.dem_geo_transform = dem_geo_transform

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
        dem_active = (dem != fillvalue)
        waterlevel_active = (waterlevel != NO_DATA_VALUE)
        active = dem_active & waterlevel_active
        depth_1d = waterlevel[active] - dem[active]

        # paste positive depths only
        negative_1d = (depth_1d <= 0)
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
                points,
                values,
                fill_value=NO_DATA_VALUE
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
        i1, i2 = h - 1 - i2, h - 1 - i1

        # note that get_nodgrid() expects a columns-first bbox
        return self.gr.cells.get_nodgrid(
            [j1, i1, j2, i2], subset_name=SUBSET_2D_OPEN_WATER,
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
            data = timeseries.only("s1", "id","coordinates","cell_coords").data
            lookup_nodes = np.full(((data["id"]).max() + 1,7), NO_DATA_VALUE)
            lookup_nodes[data["id"],0] = data["s1"]
            lookup_nodes[data["id"],1:3] = data["coordinates"].transpose()
            lookup_nodes[data["id"],3:7] = data["cell_coords"].transpose()
            self.cache[self.LOOKUP_NODES] = lookup_nodes
        return lookup_nodes

    def _get_nodgrid2(self, indices):   # _get_nodgrid with "i1, i2 = h - i2, h - i1" ipv "i1, i2 = h - 1 - i2, h - 1 - i1"
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
            [j1, i1, j2, i2], subset_name=SUBSET_2D_OPEN_WATER,
        )
    
    @property
    def lookup_lines(self):
        try:
            return self.cache[self.LOOKUP_LINES]
        except KeyError:
            lines = self.gr.lines.subset(SUBSET_2D_OPEN_WATER)
            timeseries_lines = lines.timeseries(indexes=[self.calculation_step])
            data_lines = timeseries_lines.only("au", "line_coords","line").data
            lookup_lines = np.full((len(data_lines["line"].transpose()),7), NO_DATA_VALUE)
            lookup_lines[:,0:2] = data_lines["line"].transpose()
            lookup_lines[:,2:6] = data_lines["line_coords"].transpose()
            lookup_lines[:,6:7] = data_lines["au"].transpose()
            self.cache[self.LOOKUP_LINES] = lookup_lines
        return lookup_lines

    @property
    def nr_nodes_lines(self):
        try:
            return self.cache[self.NR]
        except KeyError:
            nr_nodes = len(self.gr.nodes.subset(SUBSET_2D_OPEN_WATER).id)
            nr_lines = len(self.gr.lines.subset(SUBSET_2D_OPEN_WATER).au[-1,:])
            nr = [nr_nodes,nr_lines]
            self.cache[self.NR] = nr
        return nr

class TestCalculator(Calculator):
    def compute_nr_neighbours(index, waterlevels, au, Matrix, no_data_value):
        """ 
        Return:
            how many neighbours cell has that can be used for interpolation.
            Matrix modified for barycentric-interpolation.
            water gives waterlevels of cell and its neighbour when interpolation_type == 2.
        """
        # initialize
        neighbours = [1,1,1] 
        water = [-1,-1]
        i = -1
        k = 0

        for j in range(3):
            i = i+1
            if ((not (j==1))  and ((index[j][0].size == 0) or (waterlevels[index[j][0]] == no_data_value))) or ((j == 0) and (au[0] == 0)) or ((j==2) and (au[1] == 0)):
                # no neighbour-cell with known waterlevels or waterflow
                Matrix = np.delete(Matrix, i, 1)
                i = i-1
                neighbours[j] = 0
            elif k <2:
                water[k] = waterlevels[index[j]]
                k = k+1                
            
        return np.sum(neighbours)-1, Matrix, water

    def determine_interpolation_method(triangle_indices, waterlevels, au, Matrix, no_data_value):
        """ 
        Return:
            interpolation_type: type to be used based on the number and type of neighbours
                (0 = error, cell has no known waterlevel, should not interpolate,
                 1 = no neighbours, use waterlevel of cell
                 2 = 1 neighbour, use average waterlevel based on distance in x- or y-direction
                 3 = 2 neighbours, use barycentric coordinates resp. distance for barycentricinterpolated resp. distance_interpolated).
            water: waterlevels of cell and its neighbour when interpolation_type == 2.
            modified_Matrix: Matrix modified for barycentric-interpolation when interpolation_type == 3.
        """
        
        if waterlevels[triangle_indices[1]] == no_data_value:
            interpolation_type = 0
            print("error: interpolation type 0")
        
        nr_neighbours, modified_Matrix, water = TestCalculator.compute_nr_neighbours(triangle_indices, waterlevels, au, Matrix, no_data_value)
        
        if nr_neighbours == 0:
            interpolation_type = 1
        
        elif nr_neighbours == 1:
            interpolation_type = 2
        
        else:
            interpolation_type = 3
        return interpolation_type, water, modified_Matrix

    def average_x(points_in_subcell, nr_points, Matrix, water):
        """ 
        Return interpolated (array): waterlevels averaged based on distance in x-coordinates.
        """
        interpolated = np.zeros(nr_points)
        difx = abs(Matrix[0][0]-Matrix[0][1])
        weight1 = 1 - abs(points_in_subcell[:,0]-Matrix[0][0])/difx
        weight2 = 1 - abs(points_in_subcell[:,0]-Matrix[0][1])/difx
        interpolated = (weight1*water[0] + weight2*water[1])
        return interpolated   
    
    def average_y(points_in_subcell, nr_points, Matrix, water):
        """ 
        Return interpolated (array): waterlevels averaged based on distance in y-coordinates.
        """
        interpolated = np.zeros(nr_points)
        dify = abs(Matrix[1][0]-Matrix[1][1])
        weight1 = 1 - abs(points_in_subcell[:,1]-Matrix[1][0])/dify
        weight2 = 1 - abs(points_in_subcell[:,1]-Matrix[1][1])/dify
        interpolated = (weight1*water[0] + weight2*water[1])
        return interpolated

    def barycentric_interpolated(points_in_subcell, waterlevels, triangle_indices, triangle_vertices, au, Matrix, no_data_value):
        """ 
        Return interpolated: array of interpolated waterlevels at points in subcell, based on linear barycentric interpolation.
        Args:
            points_in_subcell: array of points to be interpolated in subcell.
            waterlevels: array of waterlevels (s1).
            triangle_indices: indices of cells whose centres are vertices of triangle used for interpolation.
            triangle_vertices: vertices of triangle used for interpolation.
            au: array of wet cross-sectional area of two neighbour-cells.
            Matrix: matrix with 1st row x-coords, 2nd row y-coords, 3rd row ones.
        """
        nr_points = len(points_in_subcell)

        interpolation_type, water, modified_Matrix = TestCalculator.determine_interpolation_method(triangle_indices, waterlevels, au, Matrix, no_data_value) 

        if interpolation_type == 0:                     # should not occur anymore
            interpolated = no_data_value*np.ones(nr_points)
            print("error: interpolation type 0")
            return interpolated

        elif interpolation_type == 1:
            interpolated = waterlevels[triangle_indices[1]]*np.ones(nr_points)
            return interpolated
 
        elif interpolation_type == 2:
            if not (modified_Matrix[0][0]-modified_Matrix[0][1] == 0):      # neighbour is to left or right (different x-, same y-coord.s)
                interpolated = TestCalculator.average_x(points_in_subcell, nr_points, modified_Matrix, water)
                return interpolated
            elif not (modified_Matrix[1][0]-modified_Matrix[1][1] == 0):    # neighbour is to bottom or top (different y-, same x-coord.s)
                interpolated = TestCalculator.average_y(points_in_subcell, nr_points, modified_Matrix, water)
                return interpolated
                
        else:
            modified_Matrix = np.matrix(modified_Matrix, dtype = 'float')
            C = np.array([waterlevels[triangle_indices[0][0][0]],waterlevels[triangle_indices[1]],waterlevels[triangle_indices[2][0][0]]])
            grid = np.c_[points_in_subcell, np.ones(len(points_in_subcell)) ].transpose()
            bary = np.dot(np.linalg.inv(modified_Matrix), grid)
            interpolated = bary.transpose()@C
                
        return interpolated

    def distance_interpolated(points_in_subcell, waterlevels, triangle_indices, triangle_vertices, au, Matrix, no_data_value):
        """ 
        Return interpolated: array of interpolated waterlevels at points in subcell, based on distance to triangle vertices
        Args:
            points_in_subcell: array of points to be interpolated in subcell.
            waterlevels: array of waterlevels (s1).
            triangle_indices: indices of cells whose centres are vertices of triangle used for interpolation.
            triangle_vertices: vertices of triangle used ofr interpolation.
            au: array of wet cross-sectional area of two neighbour-cells.
            Matrix: matrix with 1st row x-coords, 2nd row y-coords, 3rd row ones.
        """

        def distance(pt1, pt2):              # computes distance between two points
            return np.sqrt( (pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 )
        
        nr_points = len(points_in_subcell)
        interpolation_type, water, modified_Matrix = TestCalculator.determine_interpolation_method(triangle_indices, waterlevels, au, Matrix, no_data_value)

        if interpolation_type == 0:         # should not occur anymore
            interpolated = no_data_value*np.ones(nr_points)
            print("error: interpolation type 0")
            return interpolated

        elif interpolation_type == 1:
            interpolated = waterlevels[triangle_indices[1]]*np.ones(nr_points)
            return interpolated
 
        elif interpolation_type == 2:
            interpolated = np.zeros(nr_points)
            if not (modified_Matrix[0][0]-modified_Matrix[0][1] == 0):        # neighbour is to left or right (different x-, same y-coord.s)
                interpolated = TestCalculator.average_x(points_in_subcell, nr_points, modified_Matrix, water)
                return interpolated
            elif not (modified_Matrix[1][0]-modified_Matrix[1][1] == 0):      # neighbour is to bottom or top (different y-, same x-coord.s)
                interpolated = TestCalculator.average_y(points_in_subcell, nr_points, modified_Matrix, water)
                return interpolated
        
        else:
            C = np.array([waterlevels[triangle_indices[0]],waterlevels[triangle_indices[1]],waterlevels[triangle_indices[2]]])
            interpolated = np.zeros(nr_points)
            dist_pt_1 = distance(points_in_subcell,triangle_vertices[0][:])
            dist_pt_2 = distance(points_in_subcell,triangle_vertices[1][:])
            dist_pt_3 = distance(points_in_subcell,triangle_vertices[2][:])
            total_dist = 3*(dist_pt_1 + dist_pt_2 + dist_pt_3)
            frac1 = 1/dist_pt_1
            frac2 = 1/dist_pt_2
            frac3 = 1/dist_pt_3
            total_frac = frac1+frac2+frac3
            weight1 = frac1/total_frac
            weight2 = frac2/total_frac
            weight3 = frac3/total_frac
            interpolated = weight1*C[0] + weight2*C[1] + weight3*C[2]
            return interpolated
        return interpolated
             
    def find_points_subcell(points_in_cell, subcell_coords, index_points_in_cell):
        """ 
        Return:
            index_points (array): indices of the points in the subcell.
            points_in_cell[index_points_in_subcell] (array): points in the subcell.
        Args:
            points_in_subcell: array of points to be interpolated in subcell.
            subcell_coords: bottom-left and top-right coordinates of subcell.
            index_points_in_cell (array): indices of points located in this cell.
        """

        index_points_in_subcell = np.where((points_in_cell[:,0] >= subcell_coords[0]) & (points_in_cell[:,0] <= subcell_coords[2]) & (points_in_cell[:,1] >= subcell_coords[1]) & (points_in_cell[:,1] <= subcell_coords[3]))
        index_points = index_points_in_cell[index_points_in_subcell]
        return points_in_cell[index_points_in_subcell], index_points
        
    def cell_info(indexX, coords, pointscells, lines_wrt_cells, lines_au, cell_id, no_data_value):
        """ 
        Return:
            cell_to_X_coords: coordinates of cell at position X wrt original cell.
            cell_to_X_centre: centre of cell at position X.
            au_cell_to_X: wet cross-sectional area of cell at position X.
        Args:
            indexX: index of cell at position X wrt original cell.
            coords (array): bottom-left and top-right coordinates of cells.
            pointscells (array): coordinates of cel-centres.
            lines_wrt_cells (array): cells that line connects.
            lines_au (array): wet cross-sectional area at line.
            cell_id: id of original cell.
        """

        if indexX[0].size == 0:                                 # no cell at position X
            cell_to_X_coords = [no_data_value,no_data_value,no_data_value,no_data_value]
            cell_to_X_centre = [no_data_value,no_data_value]
            au_cell_to_X = 9999
        else:
            cell_to_X_coords = coords[:,indexX[0][0]]
            cell_to_X_centre = [pointscells[indexX[0][0],0],pointscells[indexX[0][0],1]]
            cell_to_X_id = indexX[0][0]+1
            line_cell_to_X = np.where(((lines_wrt_cells[0,:] == cell_to_X_id) & (lines_wrt_cells[1,:] == cell_id)) | ((lines_wrt_cells[0,:] == cell_id) & (lines_wrt_cells[1,:] == cell_to_X_id)))
            if line_cell_to_X[0].size > 0:            
                au_cell_to_X = lines_au[line_cell_to_X[0][0]]
            else:   # there is no line connecting original cell and cell at position X
                au_cell_to_X = 9999
        return cell_to_X_coords, cell_to_X_centre, au_cell_to_X

    def subcell_info(index, cell_coords, cell_centre, subcell, pointscells, indices_neighbours, cell_neighbours_centres, au_neighbours):
        """ 
        Return:
            subcell_coords: coordinates of subcell.
            triangle_indices: indices of cells whose centres are vertices of triangle used for interpolation.
            triangle_vertices: vertices of triangle used for interpolation.
            au: array of wet cross-sectional area of two neighbour-cells.
        Args:
            index = index of cell.
            cell_coords: bottom-left and top-right coordinates of cell ([x_left, y_bottom, x_right, y_top])
            cell_centre: coordinates of centre of cell.
            subcell: 1 = left-bottom, 2 = right-bottom, 3 = right-top, 4 = left-top.
            pointscells (array): coordinates of cel-centres.
            indices_neighbours (array): index of cell to left, bottom, right, top.
            cell_neighbours_centres (2x4 array): centres of cell to left, bottom, right, top.
            au_neighbours (array): wet cross-sectional area of cell to left, bottom, right, top.
        """
        [indexl, indexb, indexr, indext] = indices_neighbours
        [cell_to_left_centre, cell_to_bottom_centre, cell_to_right_centre, cell_to_top_centre] = cell_neighbours_centres
        [au_cell_to_left, au_cell_to_bottom, au_cell_to_right, au_cell_to_top] = au_neighbours

        if subcell == 1: 
            subcell_coords = [cell_coords[0],cell_coords[1],pointscells[index,0],pointscells[index,1]]
            triangle_vertices = [cell_to_left_centre,cell_centre,cell_to_bottom_centre]      # left, current, bottom
            triangle_indices = [indexl, index, indexb]
            au = [au_cell_to_left, au_cell_to_bottom]
        elif subcell == 2:
            subcell_coords = [pointscells[index,0],cell_coords[1],cell_coords[2],pointscells[index,1]]
            triangle_vertices = [cell_to_bottom_centre,cell_centre,cell_to_right_centre]     # bottom, current, right
            triangle_indices = [indexb, index, indexr]
            au = [au_cell_to_bottom, au_cell_to_right]
        elif subcell == 3: 
            subcell_coords = [pointscells[index,0],pointscells[index,1],cell_coords[2],cell_coords[3]]
            triangle_vertices = [cell_to_right_centre,cell_centre,cell_to_top_centre]        # right, current, top
            triangle_indices = [indexr, index, indext]
            au = [au_cell_to_right, au_cell_to_top]
        else: 
            subcell_coords = [cell_coords[0],pointscells[index,1],pointscells[index,0],cell_coords[3]]
            triangle_vertices = [cell_to_top_centre,cell_centre,cell_to_left_centre]         # top, current, left
            triangle_indices = [indext, index, indexl]
            au = [au_cell_to_top, au_cell_to_left]
        return subcell_coords, triangle_indices, triangle_vertices, au

    def __call__(self, indices, values, no_data_value):  
        """ 
        Return result containing interpolated waterlevels.
        Args:
            indices: indicies = (yoff, xoff), (yoff + ysize, xoff + xsize) in partition.
            values: result needs to have same shape.
            no_data_value: -9999
        """    

        # interpolating points
        points = self._get_points(indices)
        nr_points = len(points)
        pt_to_cell = self._get_nodgrid2(indices).reshape(nr_points)     # array giving for each point in which cell it is.

        # initialize result
        result = no_data_value * np.ones(nr_points)

        [max_nodes,max_lines] = self.nr_nodes_lines
        # info nodes/cells
        x = np.array([[i for i in range(1,max_nodes+1)]])
        nodes_id_s1_coords = self.lookup_nodes[x][0]
        pointscells = nodes_id_s1_coords[:,1:3]                         # cell-centre coordinates
        waterlevels = nodes_id_s1_coords[:,0]                           # waterlevels in 2D-open-water cells
        coords =  nodes_id_s1_coords[:,3:7].transpose()                 # coords of cell left-bottom corner and right-top corner 
            # [x_left, y_bottom, x_right, y_top] = [coords[0,:], coords[1,:], coords[2,:], coords[3,:]]
        
        # info lines
        x_lines = np.array([[i for i in range(0,max_lines)]])
        lines_line_linecoords_au = self.lookup_lines[x_lines][0]
        lines_wrt_cells = lines_line_linecoords_au[:,0:2].transpose()   # 2D array of cell-ids that the line connects
        lines_coords = lines_line_linecoords_au[:,2:6].transpose()      # 4D array of start and end coordinates of lines
        lines_au = lines_line_linecoords_au[:,6].transpose()            # wet cross-sectional area
        
        # obstacles (if they all block water?)
        obstacles = self.gr.lines.subset("2D_open_water_obstacles").line
        lines_with_obstacles = self.ga.lines.subset("2D_open_water_obstacles").id
        lines_au[(lines_with_obstacles-1)] = 0

        # find in which cells interpolating points are contained
        indexrange = np.sort(list(set(list(pt_to_cell))))
        if indexrange[0] == 0:
            indexrange = np.delete(indexrange,0,0)
        
        # loop over cells that contain interpolating points
        for cell_id in indexrange:
            # info cell
            index = cell_id-1
            cell_coords = coords[:,index]
            cell_centre = [pointscells[index,0],pointscells[index,1]]

            # info points in cell
            index_points_in_cell = np.where(pt_to_cell == cell_id)[0]
            points_in_cell = points[index_points_in_cell]

            # check if is waterlevel available for cell
            if waterlevels[index] == no_data_value:
                result[index_points_in_cell] = no_data_value
                continue
                
            # cell to left
            indexl = np.where((cell_coords[0] > coords[0,:]) & (cell_coords[1] == coords[1,:]) & (cell_coords[0] == coords[2,:]) & (cell_coords[3] == coords[3,:]))
            [cell_to_left_coords, cell_to_left_centre, au_cell_to_left] = TestCalculator.cell_info(indexl,coords,pointscells,lines_wrt_cells,lines_au,cell_id,no_data_value)
            
            # cell to right
            indexr = np.where((cell_coords[2] == coords[0,:]) & (cell_coords[1] == coords[1,:]) & (cell_coords[2] < coords[2,:]) & (cell_coords[3] == coords[3,:]))
            [cell_to_right_coords, cell_to_right_centre, au_cell_to_right] = TestCalculator.cell_info(indexr,coords,pointscells,lines_wrt_cells,lines_au,cell_id,no_data_value)

            # cell to bottom
            indexb = np.where((cell_coords[0] == coords[0,:]) & (cell_coords[1] > coords[1,:]) & (cell_coords[2] == coords[2,:]) & (cell_coords[1] == coords[3,:]))
            [cell_to_bottom_coords, cell_to_bottom_centre, au_cell_to_bottom] = TestCalculator.cell_info(indexb,coords,pointscells,lines_wrt_cells,lines_au,cell_id,no_data_value)

            #cell to top
            indext = np.where((cell_coords[0] == coords[0,:]) & (cell_coords[3] == coords[1,:]) & (cell_coords[2] == coords[2,:]) & (cell_coords[3] < coords[3,:]))
            [cell_to_top_coords, cell_to_top_centre, au_cell_to_top] = TestCalculator.cell_info(indext,coords,pointscells,lines_wrt_cells,lines_au,cell_id,no_data_value)
            

            sum_points = 0                                              # to check if find all points in cell via subcells
            # loop over subcell (quarter) of each cell
            for subcell in range(1,5):
                # info subcell
                indices_neighbours = [indexl,indexb,indexr,indext]
                cell_neighbours_centres = [cell_to_left_centre,cell_to_bottom_centre,cell_to_right_centre,cell_to_top_centre]
                au_neighbours = [au_cell_to_left,au_cell_to_bottom,au_cell_to_right,au_cell_to_top]
                [subcell_coords,triangle_indices,triangle_vertices,au] = TestCalculator.subcell_info(index, cell_coords, cell_centre, subcell, pointscells, indices_neighbours, cell_neighbours_centres, au_neighbours)

                # find interpolating points in each subcel
                points_in_subcell,index_points = TestCalculator.find_points_subcell(points_in_cell,subcell_coords,index_points_in_cell)
                if points_in_subcell.size == 0:                         # no points in subcell
                    continue
                sum_points = sum_points+points_in_subcell.size/2        # to check if find all points in cell via subcells

                # barycentric interpolation:
                Matrix = [[triangle_vertices[0][0],triangle_vertices[1][0],triangle_vertices[2][0]],[triangle_vertices[0][1],triangle_vertices[1][1],triangle_vertices[2][1]],[1,1,1]]
                interpolated = TestCalculator.barycentric_interpolated(points_in_subcell, waterlevels, triangle_indices, triangle_vertices, au, Matrix, no_data_value)
                
                # distance based interpolation/averaging:
                #interpolated = TestCalculator.distance_interpolated(points_in_subcell, waterlevels, triangle_indices, triangle_vertices, au, Matrix, no_data_value)
                
                result[index_points] = interpolated

            if (sum_points < points_in_cell.shape[0]):                 # to check if find all points in cell via subcells
                print(points_in_subcell.shape)
        
        result = (np.array([result])).reshape(values.shape)
        return result

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
            dem=values, fillvalue=no_data_value, waterlevel=waterlevel,
        )


class InterpolatedLevelDepthCalculator(InterpolatedLevelCalculator):
    def __call__(self, indices, values, no_data_value):
        """Return waterdepth array."""
        waterlevel = super().__call__(indices, values, no_data_value)
        return self._depth_from_water_level(
            dem=values, fillvalue=no_data_value, waterlevel=waterlevel,
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
    def __init__(self, source_path, target_path, progress_func=None):
        self.source_path = source_path
        self.target_path = target_path
        self.progress_func = progress_func

        if path.exists(self.target_path):
            raise OSError("%s already exists." % self.target_path)

    def __enter__(self):
        """Open datasets.
        """
        self.source = gdal.Open(self.source_path, gdal.GA_ReadOnly)
        block_x_size, block_y_size = self.block_size
        options = [
            "compress=deflate",
            "blockysize=%s" % block_y_size,
        ]
        if block_x_size != self.raster_x_size:
            options += [
                "tiled=yes",
                "blockxsize=%s" % block_x_size,
            ]

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
        def offset_size_range(stop, step):
            for start in range(0, stop, step):
                yield start, min(step, stop - start)

        # tiled tiff writing is much faster row-wise
        raster_size = self.raster_y_size, self.raster_x_size
        block_size = self.block_size[::-1]
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
                xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize,
            )
            indices = (yoff, xoff), (yoff + ysize, xoff + xsize)
            result = calculator(
                indices=indices,
                values=values,
                no_data_value=no_data_value,
            )

            self.target.GetRasterBand(1).WriteArray(
                array=result, xoff=xoff, yoff=yoff
            )


calculator_classes = {
    MODE_COPY: CopyCalculator,
    MODE_NODGRID: NodGridCalculator,
    MODE_CONSTANT_S1: ConstantLevelCalculator,
    MODE_INTERPOLATED_S1: InterpolatedLevelCalculator,
    MODE_CONSTANT: ConstantLevelDepthCalculator,
    MODE_INTERPOLATED: InterpolatedLevelDepthCalculator,
    MODE_TEST: TestCalculator,
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
            "dem_shape": (converter.raster_y_size, converter.raster_x_size),
        }
        with CalculatorClass(**calculator_kwargs) as calculator:
            converter.convert_using(calculator)
