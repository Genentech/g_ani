'''
Created on May 22, 2018

@author: albertgo
'''
import pytest
import numpy.testing as npt
from ase.units import Hartree, kcal
from ase.units import mol as mol_unit

from ml_qm.pt.PTMol import PTMol
from ml_qm import GDBMol as gm
from ml_qm.pt import RadialBasis as RB
from ml_qm.pt import ANIDComputer as AComput
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611


mol = None
mol2 = None
aniCompu = None


def setup_module(module): # pylint: disable=W0613
    """ setup any state specific to the execution of the given module."""
    global mol, mol2, aniCompu # pylint: disable=W0603
    mol  = PTMol(gm.GDBMol(gm.demoMols['O']))
    mol2 = PTMol(gm.GDBMol(gm.demoMols['C#C']))
    rbasis   = RB.GaussianRadialBasis([1,6],5,1,5,None,2,5) # large cutoff removes asymmetry
    aniCompu = AComput.ANIDComputer([1,6], rbasis)




def testDistMatrix():
    global mol # pylint: disable=W0603

    dm, i, j = mol.neigborDistanceLT(3)
    npt.assert_allclose(dm.numpy(), [1, 1, 1.41421], 1e-4)
    npt.assert_array_equal(i, [1,2,2])
    npt.assert_array_equal(j, [0,0,1])

    dm, i, j = mol.neigborDistance(3)
    npt.assert_allclose(dm.numpy(), [1, 1, 1, 1.41421, 1, 1.41421], 1e-4)
    npt.assert_array_equal(i, [0,0,1,1,2,2])
    npt.assert_array_equal(j, [1,2,0,2,0,1])


def testNuclearRepulsion():
    molecul  = PTMol(gm.GDBMol(gm.demoMols['H2O_HF']))

    # value from gaussian
    assert molecul.atomizationE == pytest.approx(7.4171, 1e-4)
    assert molecul.nuclearRepulsion / Hartree/mol_unit*kcal == pytest.approx(9.3467, 1e-4)

    assert molecul.elctronicAtomizationE + molecul.nuclearRepulsion == pytest.approx(molecul.atomizationE)
