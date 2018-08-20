#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
import pypandoc

__version__ = '0.0.1'

long_description = pypandoc.convert_file('README.md', 'rst')

required = [
    'ipython==6.5.0',
    'numpy==1.15.0',
    'pandas==0.23.3',
    'pytest==3.7.1',
    'scipy==1.1.0',
    'pypandoc==1.4.0',
]

kwargs = {
    "name": "preposterous",
    "version": str(__version__),
    "packages": ["preposterous"],
    "description": "A simple library for estimating the impact of an intervention, with humility",
    "long_description": long_description,
    "author": "Matthew Ritter",
    "maintainer": "Matthew Ritter",
    "license": "MIT",
    "url": "https://github.com/matthewwritter/preposterous",
    "keywords": "quantified_self statistics",
    "classifiers": [
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ]
}

# install_requires treated separately for PyCharm introspection
setup(**kwargs, install_requires=required)
