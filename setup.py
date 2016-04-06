#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import glob

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

setup(
    name="prospect",
    url="https://github.com/bd-j/prospect",
    version='0.1.0',
    author="Ben Johnson",
    author_email="benjamin.johnson@cfa.harvard.edu",
    packages=["prospect",
              "prospect.models",
              "prospect.likelihood",
              "prospect.fitting",
              "prospect.sources",
              "prospect.io",
              "prospect.utils"],

    license="LICENSE",
    description="Stellar Population Inference",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    scripts=glob.glob("scripts/*.py"),
    include_package_data=True,
    install_requires=["numpy"],
)
