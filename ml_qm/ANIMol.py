# Alberto
# pylint: disable=W0105

from ase.atoms import Atoms
from ase.units import Hartree, mol, kcal
from cdd_chem.util.io import warn # noqa: F401; $ pylint: disable=W0611


class ANIMol():

    """ from: https://github.com/isayev/ASE_ANI/blob/master/ani_models/ani-1x_8x/sae_linfit.dat """
    atomEnergies_pre_201904 = {   # [kcal/mol]
        1:   -0.60095298000 * Hartree*mol/kcal,
        6:  -38.08316124000 * Hartree*mol/kcal,
        7:  -54.70775770000 * Hartree*mol/kcal,
        8:  -75.19446356000 * Hartree*mol/kcal,
    }

    atomEnergies = {   # from ANI2 model   # [kcal/mol]
        1: -0.5975740895940036 * Hartree*mol/kcal,
        6: -38.08963824205658  * Hartree*mol/kcal,
        7: -54.71039678680539  * Hartree*mol/kcal,
        8: -75.19115551439664  * Hartree*mol/kcal,
        9: -99.79839227891235  * Hartree*mol/kcal,
        15: -340.              * Hartree * mol / kcal, # for now just random
        16: -398.1787966261814 * Hartree * mol / kcal,
        17:-460.206968972609   * Hartree*mol/kcal,
    }

    atomEnergies202001 = {  #from ANI2 202001, determined with ANI_get_info.py
        1:   -0.5960826077395001 * Hartree*mol/kcal,
        6:   -38.09742369135185  * Hartree*mol/kcal,
        7:   -54.72359774006924  * Hartree*mol/kcal,
        8:   -75.19261984417626  * Hartree*mol/kcal,
        9:   -99.79121500670908  * Hartree*mol/kcal,
        16:  -398.13769382832004 * Hartree*mol/kcal,
        17:  -460.15600917560806 * Hartree*mol/kcal,
    }


    khanAtomEnergies = {   # [kcal/mol]
        1:  -0.499321232710* Hartree*mol/kcal,
        6:  -37.8338334397 * Hartree*mol/kcal,
        7:  -54.5732824628 * Hartree*mol/kcal,
        8:  -75.0424519384 * Hartree*mol/kcal,
    }

    """ old 
    atomEnergies = {   # [kcal/mol]
        1:   -0.500273 * Hartree*mol/kcal,
        6:  -37.846772 * Hartree*mol/kcal,
        7:  -54.583861 * Hartree*mol/kcal,
        8:  -75.064579 * Hartree*mol/kcal,
        9:  -99.718730 * Hartree*mol/kcal,
    }
    """

    """  old incorrect values
    atomEnergies = {   
        1:   -0.500607632585 * Hartree*mol/kcal,
        6:  -37.8302333826 * Hartree*mol/kcal,
        7:  -54.5680045287 * Hartree*mol/kcal,
        8:  -75.064579 * Hartree*mol/kcal,
        9:  -99.0362229210 * Hartree*mol/kcal,
    }
    """

    def __init__(self, name, energy, atNums, xyz):
        self.energy_kcal = energy
        self.name = name
        self.atNums = list(atNums)
        self.nAt = len(self.atNums)
        self.atoms = Atoms(numbers=self.atNums, positions=xyz)


    @property
    def nHeavy(self):
        return sum(1 for an in self.atoms.numbers if an>1)

    @property
    def energy(self):
        """ Enthalpy at 0K in kcal/mol """

        return self.energy_kcal


    @property
    def atomizationE(self):
        return self.energy - sum( [ ANIMol.atomEnergies[atNum] for atNum in self.atNums ])

    @property
    def xyz(self):
        return self.atoms.get_positions()
