#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
import pypandoc
import preposterous

long_description = pypandoc.convert_file('README.md', 'rst')

required = [
    'ipython==6.5.0',
    'numpy==1.15.0',
    'pandas==0.23.3',
    'pytest==3.7.1',
    'scipy==1.1.0',
    'pypandoc==1.4.0',
    'matplotlib>=2.2.2'
]

kwargs = {
    "name": "preposterous",
    "version": preposterous.__version__,
    "packages": ["preposterous"],
    "description": "A simple library for estimating the impact of an intervention, with humility",
    "long_description": long_description,
    "author": "Matthew Ritter",
    "maintainer": "Matthew Ritter",
    "license": "MIT",
    "url": "https://github.com/matthewwritter/preposterous",
    "classifiers": [
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",

    ],
    'tests_require': ['pytest'],
    'extras_require': {
        'testing': ['pytest'],
    },
    'keywords': ['quantified self', 'quantifiedself', 'statistics', 'bayesian statistics', 'time series'],


}

# install_requires treated separately for PyCharm introspection
setup(**kwargs, install_requires=required)
