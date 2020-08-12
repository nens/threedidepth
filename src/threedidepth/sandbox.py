# -*- coding: utf-8 -*-

from threedigrid.admin.gridresultadmin import GridH5ResultAdmin


GRIDADMIN_PATH = "var/testdata/12997/gridadmin.h5"
RESULTS_3DI_PATH = "var/testdata/12997/results_3di.nc"


def skirmish():
    f = GRIDADMIN_PATH
    nc = RESULTS_3DI_PATH
    # init gives warning:
    # [!] threedicore version differ!
    # Version result file has been created with: b'2.0.9'
    # Version gridadmin file has been created with: 1.4.20-1

    # context manager returns only underlying h5 file
    with GridH5ResultAdmin(f, nc) as gr:
        assert not isinstance(gr, GridH5ResultAdmin)

    # using .value instead of slicing gives deprecation warnings

    # otherwise files are kept open
    gr = GridH5ResultAdmin(f, nc)
    assert isinstance(gr, GridH5ResultAdmin)

    # attribute error
    # cs = gr.get_timeseries_chunk_size()
    cs = gr.timeseries_chunk_size  # noqa

    # threedigrid does not seem to support snap-to-timesep?
    qs = gr.nodes.timeseries(start_time=150, end_time=150)
    assert len(qs.s1) == 0

    # We'll have to resort to indexed steps. Negative works, slices too.
    qs = gr.nodes.timeseries(indexes=[-4, -3, -2, -1])
    assert len(qs.s1) == 4

    # now what, via the bounds?
    gr.cells.subset('2D_OPEN_WATER').bounds  # ?
    # Now, we need the layout of the nodes
    # Read the projection

    # or get the node grid directly - but it is big
    dem_shape = 29125, 8223
    dem_pixelsize = 0.5
    gr.grid.get_pixel_map(dem_pixelsize, dem_shape)

    return gr


skirmish()
