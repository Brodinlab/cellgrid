#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

requirements = [
    'fcsy>=0.3.0',
    'numpy>=1.15.4',
    'pandas>=0.24.2',
    'scikit-learn',
    'xgboost>=0.9.0'
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="YC",
    author_email='yang.chen@scilifelab.se',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Cell classification by learning known phenotypes",
    install_requires=requirements,
    license="MIT license",
    long_description='',
    include_package_data=True,
    keywords='cellgrid',
    name='cellgrid',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Brodinlab/cellgrid',
    version='0.5.6',
    zip_safe=False,
)
