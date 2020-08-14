threedidepth
============

Calculate waterdepths for 3Di results.

Features:

* Interpolated or gridcell-constant waterlevels
* Interfaces with threediresults via `threedigrid`
* Progress indicator support
* Low memory consumption


usage
-----

$ threedidepth gridadmin.h5 results_3di.nc dem.tif waterdepth.tif

>>> threedidepth.calculate_waterdepth(...)
