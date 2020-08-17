threedidepth
============

Calculate waterdepths for 3Di results.

* Interpolated or gridcell-constant waterlevels
* Interfaces with threediresults via `threedigrid`
* Progress indicator support
* Low memory consumption


Installation
------------

Make sure GDAL is available as (`from osgeo import gdal`)

$ pip install threedidepth  # TODO, we're not yet on pypi


Usage
-----

$ threedidepth gridadmin.h5 results_3di.nc dem.tif waterdepth.tif

>>> threedidepth.calculate_waterdepth(...)


Development installation with docker-compose
--------------------------------------------

For development, you can use a docker-compose setup::

    $ docker-compose build --build-arg uid=`id -u` --build-arg gid=`id -g` lib
    $ docker-compose up --no-start
    $ docker-compose start
    $ docker-compose exec lib bash

(Re)create & activate a virtualenv::

    (docker)$ rm -rf .venv
    (docker)$ virtualenv .venv --system-site-packages
    (docker)$ source .venv/bin/activate

Install dependencies & package and run tests::

    (docker)(virtualenv)$ pip install -r requirements.txt
    (docker)(virtualenv)$ pip install -e .[test]
    (docker)(virtualenv)$ pytest

Update requirements.txt::
    
    (docker)$ deactivate
    (docker)$ rm -rf .venv
    (docker)$ virtualenv .venv
    (docker)$ source .venv/bin/activate
    (docker)(virtualenv)$ pip install .
    (docker)(virtualenv)$ pip install pygdal==2.2.3.*
    (docker)(virtualenv)$ pip uninstall raster-store --yes
    (docker)(virtualenv)$ pip freeze > requirements.txt


