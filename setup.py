#!/usr/bin/env python
"""
(C) 2021 Genentech. All rights reserved.

The setup script.
"""

import ast
import os
import re

from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

# requirements are defined in the conda package level (see conda/meta.yaml.in)

requirements = []
setup_requirements = []

VERSION = None
abspath = os.path.dirname(os.path.abspath(__file__))
version_file_name = os.path.join(abspath, "ml_qm", "__init__.py")
with open(version_file_name) as version_file:
    version_file_content = version_file.read()
    version_regex = re.compile(r"__version__\s+=\s+(.*)")
    match = version_regex.search(version_file_content)
    assert match, "Cannot find version number (__version__) in {}".format(version_file_name)
    VERSION = str(ast.literal_eval(match.group(1)))

setup(
    author="Gobbi, Alberto",
    author_email='Gobbi.Alberto@gene.com',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
    ],
    description=" ml_qm contains the Genentech Neural Net Potential Code",
    entry_points={
        'console_scripts': [
            'sdfGeometric.py=ml_qm.optimize.geometric.SDFGeomeTRIC:main',
            'sdfMOptimizer.py=ml_qm.optimize.SDFMOptimizer:main',
            'sdfNNP.py=ml_qm.optimize.SDF_NNP:main',
            'sdfNNPKa.py=ml_qm.pKaNet.SDF_NN_PKa:main',
            'createDistNetBatchDataset.py=ml_qm.distNet.batch_data_set:main',
            'trainMLQM.py=scripts.trainMem:main'
        ],
    },
    scripts=['scripts/sdfNNPConfAnalysis.pl', 'scripts/sdfNNPConfSample.pl'],
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/x-rst',
    include_package_data=True,
    keywords='ml_qm',
    name='ml_qm',
    packages=find_packages(include=['ml_qm', 'ml_qm.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=['pytest', 'scripttest'],
    version=VERSION,   # please update version number in "ml_qm"/__init__.py file
    zip_safe=False,
)
