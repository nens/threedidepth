# -*- coding: utf-8 -*-
from itertools import product

import numpy as np
from scipy import ndimage


class Pairs:
    """
    Group pairs of cells.

    TODO this naming may be a bit obscure. See if we can make separate
    functions for it. Or a generic function accepting defined constant
    parameters on the CornerCalculator.
    """
    def __init__(self, array):
        a, b, c, d = array.transpose()
        # clockwise pairs, starting at the top
        self.cw = (a, b), (b, d), (d, c), (c, a)
        # clockwise pairs, starting at the bottom, "opposite"
        self.op = (d, c), (c, a), (a, b), (b, d)
        # 3 pairs cw, starting at top, then 3 pairs ccw, start left, "double 3"
        self.d3 = (a, b), (b, d), (d, c), (a, c), (c, d), (d, b)


class Linked:
    def __init__(self, lines):
        self.links = np.empty((len(lines[0]), 2), dtype="i8")
        self.view = self.links.view("i8, i8")
        self.links[:] = lines.transpose()

    def __getitem__(self, index):
        a, b = index
        array = np.empty((len(a), 2), dtype="i8")
        view = array.view("i8, i8").ravel()
        array[:, 0], array[:, 1] = a, b
        result = np.isin(view, self.view)
        array[:, 0], array[:, 1] = b, a
        result |= np.isin(view, self.view)
        return result


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
        linked: something that checks if nodes are linked...
        """
        _nodes = self._nodes
        no_node = self._no_node
        assert values[no_node] == no_value

        # values[1:] = np.log10(np.arange(1, values.size))  # for debugging
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
            match = (
                (l1 != no_label) &
                (l2 != no_label) &
                ((n1 == n2) | linked[n1, n2])
            )
            # stop label becomes start label
            l2[match] = l1[match]

        # calculate corner values
        index = np.unique(_labels)
        means = np.full(no_label + 1, no_value)
        means[index] = ndimage.mean(_values, _labels, index)
        _corners = means[_labels]

        # create result, but write only corners from 'no T-bar' intersections
        result = np.full((values.size, 4), no_value)
        nobar = ~bar.any(axis=0)
        i = _nodes[nobar].ravel()
        j = 3 - np.arange(4 * nobar.sum()) % 4
        result[i, j] = _corners[nobar].ravel()

        # groups containing a T-bar get the mean of the adjacent corners
        result_p = Pairs(result)
        all_pairs = zip(bar, _labels_p.cw, _nodes_p.cw, result_p.op)
        for b, (l1, l2), (n1, n2), (r3, r4) in all_pairs:
            means[l1[b]] = 0.5 * (r3[n1[b]] + r4[n2[b]])
            # deactivate the labels, so that they do not overwrite results
            l1[b] = no_label
            l2[b] = no_label

        # write the corners again, now with correct 'T-bar' intersections
        _corners = means[_labels]

        # write result for all active corners
        active = (_labels != no_label)
        i = _nodes[active]           # node index
        j = 3 - active.nonzero()[1]  # corner index, opposite to intersection
        result[i, j] = _corners[active]

        return result


class BilinearInterpolator:
    """
    nodes: the full nodgrie
    no_node: Value in the nodgrid indicating no value
    values: values per node
    no_value: value indicating no value
    edges; per node edges, x1, y1, x2, y2
    """
    def __init__(self, nodes, no_node, values, no_value, edges, linked):
        self.corners = CornerCalculator(
            nodes=nodes, no_node=no_node,
        ).get_corners(
            values=values, no_value=no_value, linked=linked,
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


# TODO more tests in a separate test module
NODES = np.array([
    [0, 1, 4, 4],
    [2, 3, 4, 4],
])
NO_NODE = 0
VALUES = np.arange(5.)
NO_VALUE = 9.
VALUES[NO_NODE] = NO_VALUE
LINES = np.array([
    [1, 2],
    # [1, 4],
    [2, 3],
    [3, 4],
    [1, 3],
]).transpose()


def test():
    calculator = CornerCalculator(nodes=NODES, no_node=NO_NODE)
    print(calculator._nodes.reshape(-1, 2, 2))

    corners = calculator.get_corners(
        values=VALUES, no_value=NO_VALUE, linked=Linked(LINES)
    )
    print(corners.reshape(-1, 2, 2))


# test()
