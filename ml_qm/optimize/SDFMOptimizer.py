#!/usr/bin/env python
# encoding: utf-8

'''
Wrapper around sdf_ulti_optimize.py to set our own NNPComputerFactory

@author:     albertgo

@copyright:  2019 Genentech Inc.

'''

import sys
from t_opt import sdf_multi_optimizer
from ml_qm.optimize.NNP_computer_factory import NNPComputerFactory


def main():
    sdf_multi_optimizer.main(NNPComputerFactory)


if __name__ == "__main__":
    sys.exit(main())
