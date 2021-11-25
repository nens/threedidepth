# -*- coding: utf-8 -*-
from itertools import product

import numpy as np
from scipy import ndimage


class Pairs:
    def __init__(self, array):
        a, b, c, d = array.transpose()
        self.cw = (a, b), (b, d), (d, c), (c, a)
        self.op = (d, c), (c, a), (a, b), (b, d)
        self.d3 = (a, b), (b, d), (d, c), (a, c), (c, d), (d, b)


class CornerCalculator:
    def __init__(self, nodes, no_node):
        """
        nodes: 2D node grid
        no_node: value indicating not in a node
        """
        self._nodes = self.get_intersections(nodes=nodes, no_node=no_node)
        self._no_node = no_node

    def get_intersections(self, nodes, no_node):
        """
        Return array that lists nodes per intersection.
        """
        height, width = nodes.shape

        # generate index tuple for each corner of cell ('object') in nodes
        corners = (
            c
            for node_id, obj in enumerate(ndimage.find_objects(nodes + 1))
            if obj is not None and node_id != no_node
            for c in product(*((s.start, s.stop) for s in obj))
        )

        # collect all unique 2 x 2 node array around a corner as tuple
        intersections = set()

        for i0, j0 in corners:
            intersections.add(tuple(
                nodes[i0 + di, j0 + dj]
                if 0 <= i0 + di < height and 0 <= j0 + dj < width
                else no_node
                for di, dj in ((-1, -1), (-1, 0), (0, -1), (0, 0))
            ))

        return np.array(list(intersections))

    def get_corners(self, values, no_value, linked):
        """
        Return corner values per node.

        values: node values
        no_value: value indicating no value
        linked: scipy.sparse.SPMmatrix
        """
        _nodes = self._nodes
        no_node = self._no_node
        assert values[no_node] == no_value

        _values = values[_nodes]

        # make pairs of nodes that go around an intersection, order matters!
        _nodes_p = Pairs(_nodes)

        # determine T-bar intersections per pair
        bar = np.zeros(_nodes.shape[::-1], dtype=bool)
        for i, (n1, n2) in enumerate(_nodes_p.cw):
            bar[i] = (n1 == n2) & (n1 != no_node)

        # give every active corner a unique label, and make pairs too
        no_label = _nodes.size
        _labels = np.arange(no_label).reshape(_nodes.shape)
        _labels[_values == no_value] = no_label
        _labels_p = Pairs(_labels)

        # put connected nodes in the same group
        for (n1, n2), (l1, l2) in zip(_nodes_p.d3, _labels_p.d3):
            # determine active links with active endpoints
            match = (l1 != no_label) & (l2 != no_label)  # & linked[n1, n2]
            # stop label becomes start label
            l2[match] = l1[match]

        # calculate corner values and make pairs again
        means = ndimage.mean(_values, _labels, range(no_label + 1))
        assert means[no_label] == no_value
        _corners = means[_labels]

        # create result, but write only corners from 'no T-bar' intersections
        result = np.full((values.size, 4), no_value)
        nobar = ~bar.any(axis=0)
        i = _nodes[nobar].ravel()
        j = 3 - np.arange(4 * nobar.sum()) % 4
        result[i, j] = _corners[nobar].ravel()

        # make corner pairs that start on opposite side of intersection
        result_p = Pairs(result)

        # groups containing a T-bar get the mean of the adjacent corners
        all_pairs = zip(bar, _labels_p.cw, _nodes_p.cw, result_p.op)
        for b, (l1, l2), (n1, n2), (r3, r4) in all_pairs:
            # adjust the mean TODO somehow wrong mean gets calculated. earlier
            # we used _corners instead of results, but how would that be
            # correct?
            means[l1[b]] = 0.5 * (r3[n1[b]] + r4[n2[b]])
            # deactivate the labels
            l1[b] = no_label
            l2[b] = no_label

        # denormalize the corners per node into the result, but make sure not
        # to overwrite the already written 'no T-bar' intersections, since
        # those have already been written before 
        result = np.full((values.size, 4), no_value)  #  TODO nice to have would be to
        # only write the T-bar intersections, and only the active labels in
        # them we don't have to recreate the result array then
        active = (_labels != no_label)
        i = _nodes[active]           # node index
        j = 3 - active.nonzero()[1]  # corner index, opposite to intersection
        result[i, j] = _corners[active]

        import pdb
        pdb.set_trace() 
        return result


class BilinearInterpolator:
    """
    nodes: the full nodgrie
    no_node: Value in the nodgrid indicating no value
    values: values per node
    no_value: value indicating no value
    edges; per node edges, x1, y1, x2, y2
    """
    def __init__(self, nodes, no_node, values, no_value, edges):
        self.corners = CornerCalculator(
            nodes=nodes, no_node=no_node,
        ).get_corners(
            values=values, no_value=no_value, linked=None,
        )
        self.edges = edges

    def __call__(self, nodes, points):
        """ local nodes and points """
        c12, c22, c11, c21 = self.corners[nodes].transpose()
        x1, y1, x2, y2 = self.edges[:, nodes]
        x, y = points.transpose()
        result = np.sum([
            c11 * (x2 - x) * (y2 - y) / (x2 - x1) / (y2 - y1),
            c12 * (x2 - x) * (y - y1) / (x2 - x1) / (y2 - y1),
            c21 * (x - x1) * (y2 - y) / (x2 - x1) / (y2 - y1),
            c22 * (x - x1) * (y - y1) / (x2 - x1) / (y2 - y1),
        ], axis=0)
        return result
