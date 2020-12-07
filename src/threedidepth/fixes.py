# -*- coding: utf-8 -*-

import h5py
import numpy as np
from threedigrid.admin.nodes.subsets import NODE_TYPE__IN_SUBSETS


def fix_gridadmin(gridadmin_path):
    """Fix older type gridadmin files.
    """
    with h5py.File(gridadmin_path, "r") as h5py_file:
        node_group = h5py_file["nodes"]
        if "pixel_coords" in node_group:
            return

    with h5py.File(gridadmin_path, "r+") as h5py_file:
        nodk = h5py_file["grid_coordinate_attributes"]["nodk"].value
        nodm = h5py_file["grid_coordinate_attributes"]["nodm"].value
        nodn = h5py_file["grid_coordinate_attributes"]["nodn"].value
        ip = h5py_file["grid_coordinate_attributes"]["ip"].value
        jp = h5py_file["grid_coordinate_attributes"]["jp"].value
        node_types = h5py_file["nodes"]["node_type"].value
        pixel_coords = np.full((4, nodk.shape[0]), -9999, dtype="int")
        for node_type_subset in NODE_TYPE__IN_SUBSETS["2D_ALL"]:
            mask = node_types == node_type_subset
            pixel_coords[0, mask] = ip[0, nodm[mask] - 1, nodk[mask] - 1] - 1
            pixel_coords[1, mask] = jp[0, nodn[mask] - 1, nodk[mask] - 1] - 1
            pixel_coords[2, mask] = ip[3, nodm[mask] - 1, nodk[mask] - 1]
            pixel_coords[3, mask] = jp[3, nodn[mask] - 1, nodk[mask] - 1]
        node_group = h5py_file["nodes"]
        node_group.create_dataset(
            "pixel_coords", data=pixel_coords, dtype="int"
        )
