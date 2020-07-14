from setuptools import setup

version='0.1.dev0'

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('CHANGES.rst') as changes_file:
    changes = changes_file.read()

install_requires = [
    "threedigrid",
    "gdal",
    "numpy",
    "numba",  # missing dependency in threedigrid 1.0.20.9
    "h5py",  # explicit because of the gridadmin fix
]

tests_require = ["flake8", "ipdb", "ipython", "pytest", "pytest-cov"]

setup(
    name='threedidepth',
    version=version,
    description="Calculate waterdepths for 3Di results.",
    long_description=readme + '\n\n' + changes,
    author="Arjan Verkerk",
    author_email='arjan.verkerk@nelen-schuurmans.nl',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    license="BSD license",
    keywords=['threedidepth'],
    packages=["threedidepth"],
    test_suite='tests',
    url='https://github.com/nens/threedidepth',
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require={"test": tests_require},
)
