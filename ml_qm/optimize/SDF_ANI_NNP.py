#!/usr/bin/env python
# encoding: utf-8

'''
Wrapper around SDF_NNP.py to load pyNeuroChem first.
Loading pyNeuroChem after pytorch causes a not understood crash

@author:     albertgo

@copyright:  2019 Genentech Inc.

'''

import pyNeuroChem as neuro   # noqa: F401; # pylint: disable=W0611,E0401
import sys
import ml_qm.optimize.SDF_NNP as SDF_NNP



if __name__ == "__main__":
    sys.exit(SDF_NNP.main())
