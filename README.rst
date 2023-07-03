threedidepth
============

Calculate waterdepths for 3Di results. For results of type 'raw' the variable
's1' is used as waterlevel. For results of type 'aggregate', the variable
's1_max' is used as waterlevel.

* Interpolated or gridcell-constant waterlevels
* Interfaces with threediresults via `threedigrid`
* Progress indicator support
* Low memory consumption

For the interpolated mode, the 'lizard'-method is used. For a detailed
description, read the docstring for the `LizardLevelCalculator`.

For the maximum waterlevel calculation, the maximum waterlevel for each point
is taken before the interpolation is applied. This can lead to situations where
the highest waterlevel for a pixel for a certain timestep is higher than the
maximum waterlevel for the pixel.


Installation
------------

Make sure GDAL is available as (`from osgeo import gdal`)

$ pip install threedidepth


Usage
-----

From the cli::

    $ threedidepth gridadmin.h5 results_3di.nc dem.tif waterdepth.tif


Or python::

    >>> threedidepth.calculate_waterdepth(...)


Development installation with Docker Compose
--------------------------------------------

For development, clone the repository and use a docker compose setup::

    $ docker compose build --build-arg uid=`id -u` --build-arg gid=`id -g` lib
    $ docker compose up --no-start
    $ docker compose start
    $ docker compose exec lib bash

Create a virtualenv::

    # note that Dockerfile prepends .venv/bin to $PATH
    (docker)$ virtualenv .venv --system-site-packages

Install dependencies & package and run tests::

    (docker)$ pip install -r requirements.txt
    (docker)$ pip install -e .[test]
    (docker)$ pytest

Update packages::
    
    (docker)$ rm -rf .venv
    (docker)$ virtualenv .venv --system-site-packages
    (docker)$ pip install -e .
    (docker)$ pip freeze | grep -v threedidepth > requirements.txt
