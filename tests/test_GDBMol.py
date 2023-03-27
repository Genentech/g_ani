# pylint: disable=W0603
'''
Created on May 22, 2018

@author: albertgo
'''

import pytest
from ml_qm import GDBMol as gm


mol = None

def setup_module(module):
    """ setup any state specific to the execution of the given module."""
    global mol
    mol = gm.GDBMol(gm.demoMols['C'])



def test_energy():
    global mol

    assert -395.99989364979774 == pytest.approx(mol.atomizationE)
