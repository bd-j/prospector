#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import glob
#import subprocess
try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

#githash = subprocess.check_output(["git", "log", "--format=%h"], universal_newlines=True).split('\n')[0]
vers = "1.1.0"
githash = ""
with open('prospect/_version.py', "w") as f:
    f.write('__version__ = "{}"\n'.format(vers))
    f.write('__githash__ = "{}"\n'.format(githash))

setup(
    name="astro-prospector",
    version=vers,
    project_urls={"Source repo": "https://github.com/bd-j/prospector",
                  "Documentation": "https://prospect.readthedocs.io"},
    author="Ben Johnson",
    author_email="benjamin.johnson@cfa.harvard.edu",
    classifiers=["Development Status :: 5 - Production/Stable",
                 "Intended Audience :: Science/Research",
                 "Programming Language :: Python",
                 "License :: OSI Approved :: MIT License",
                 "Natural Language :: English",
                 "Topic :: Scientific/Engineering :: Astronomy"],
    packages=["prospect",
              "prospect.data",
              "prospect.likelihood",
              "prospect.fitting",
              "prospect.models",
              "prospect.sources",
              "prospect.io",
              "prospect.plotting",
              "prospect.utils"],
    license="MIT",
    description="Stellar Population Inference",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    package_data={"": ["README.md", "LICENSE"]},
    scripts=glob.glob("scripts/*.py"),
    include_package_data=True,
    install_requires=["numpy", "h5py"],
)
