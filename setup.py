# -*- encoding: utf-8 -*-
from glob import glob
from os.path import basename
from os.path import splitext
from setuptools import find_packages
from setuptools import setup

version = '0.6.3'

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('CHANGES.rst') as changes_file:
    changes = changes_file.read()

install_requires = [
    "h5py",
    "h5netcdf",
    "numpy",
    "scipy",
    "threedigrid",
    # "gdal",  # assumed available via virtualenv --system-site-packages
]

tests_require = ["flake8", "pytest", "pytest-cov"]


setup(
    name='threedidepth',
    description="Calculate waterdepths for 3Di results.",
    version=version,
    author="Arjan Verkerk",
    author_email='arjan.verkerk@nelen-schuurmans.nl',
    license="BSD license",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        'Topic :: Scientific/Engineering',
    ],
    long_description=readme + '\n\n' + changes,
    python_requires='>=3.5',
    keywords=['threedidepth'],
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    test_suite='test',
    url='https://github.com/nens/threedidepth',
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require={"test": tests_require},
    entry_points={
        'console_scripts': [
            'threedidepth=threedidepth.commands:threedidepth',
        ],
    },
)
