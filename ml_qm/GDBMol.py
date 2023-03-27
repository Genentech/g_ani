# Alberto
"""
for reading files from: 
    Ramakrishnan, Raghunathan, Pavlo O. Dral, Matthias Rupp, and O. Anatole von Lilienfeld. 
    Quantum Chemistry Structures and Properties of 134 Kilo Molecules.
    Scientific Data 1 (August 5, 2014): 140022. https://doi.org/10.1038/sdata.2014.22.

    All properties are converted to kcal/mol and A if possible
"""

import numpy as np
import re
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611
from ase import Atoms
from ase.units import Hartree, Bohr, Ang, mol, kcal
from ase.neighborlist import neighbor_list
import ml_qm.AtomInfo as AtomInfo

import cdd_chem.util.string as string


class GDBMol():
    """ molecule from GDB """
    prop_names = ['rcA', 'rcB', 'rcC', 'mu', 'alpha', 'homo', 'lumo',
                  'gap', 'r2', 'zpve',
                  'energy_U0', 'energy_U', 'enthalpy_H',
                  'free_G', 'Cv']

    atomEnergies = {   # [kcal/mol]
        1:   -0.500273 * Hartree*mol/kcal,
        6:  -37.846772 * Hartree*mol/kcal,
        7:  -54.583861 * Hartree*mol/kcal,
        8:  -75.064579 * Hartree*mol/kcal,
        9:  -99.718730 * Hartree*mol/kcal,
    }

    _conversions = [1., 1., 1., 1., Bohr ** 3 / Ang ** 3, Hartree*mol/kcal, Hartree*mol/kcal,
                    Hartree*mol/kcal, Bohr**2/Ang**2, Hartree*mol/kcal,
                    Hartree*mol/kcal, Hartree*mol/kcal, Hartree*mol/kcal,
                    Hartree*mol/kcal, 1.]

    @staticmethod
    def _parseAtLine(lin):
        lin = lin.replace("*^","e")
        splt= lin.split()
        return [float(c) for c in splt[1:4]]


    def __init__(self, molStr, quiet=False):
        self.strg = molStr
        lines= molStr.splitlines()

        self.nAt   = int(lines[0])
        xyz   = np.array([GDBMol._parseAtLine(lin) for lin in lines[2:self.nAt+2] ])
        self.atNums = [AtomInfo.NameToNum[lin.split()[0]] for lin in lines[2:self.nAt+2]]
        self.atoms = Atoms(numbers=self.atNums,positions=xyz)

        # read properties
        self.properties = {}
        splt = lines[1].split("\t")
        self.name = splt[0]
        for pName, pVal, conv in zip(self.prop_names, splt[1:], self._conversions):
            self.properties[pName] = float(pVal) * conv

        self.smilesOK = True
        self.neighborsStruct = None

        if not quiet:
            splt = lines[self.nAt+3].split()
            try:
                #smi1 = Chem.MolToSmiles(Chem.MolFromSmiles(splt[0]),isomericSmiles=False)
                #smi2 = Chem.MolToSmiles(Chem.MolFromSmiles(splt[1]),isomericSmiles=False)
                #if smi1 != smi2:
                #    if not quiet:
                #        warn("Smiles differ for %s (%s != %s)" % (self.name, smi1, smi2))
                #    self.smilesOK = False
                pass
            except:            # noqa: E722; # pylint: disable=W0702
                if not quiet:
                    warn("Smiles parsing error for %s (%s ===== %s)" % (self.name, *splt))
                self.smilesOK = False

            (inchi1, inchi2) = lines[self.nAt+4].split()
            if not inchi2.startswith(inchi1):
                if not quiet:
                    warn("inchi differ for %s (%s != %s)" % (self.name, inchi1, inchi2))


    @property
    def nHeavy(self):
        return sum(1 for an in self.atoms.numbers if an>1)

    @property
    def energy(self):
        """ Enthalpy at 0K in kcal/mol """

        return self.properties['energy_U0']


    @property
    def atomizationE(self):
        """ kcal/mol"""
        return self.energy - sum( [ self.atomEnergies[atNum] for atNum in self.atoms.numbers ])


    @property
    def xyz(self):
        return self.atoms.get_positions()


    def distMatrixOld(self, cutoff):
        """
        >>> mol =GDBMol(demoMol['C'])
        >>> mol.distMatrix(3)
        array([[0.       , 1.091953 , 1.0919516, 1.0919464, 1.0919476],
               [1.091953 , 0.       , 1.7831198, 1.7831475, 1.7831566],
               [1.0919516, 1.7831198, 0.       , 1.7831577, 1.7831484],
               [1.0919464, 1.7831475, 1.7831577, 0.       , 1.7831479],
               [1.0919476, 1.7831566, 1.7831484, 1.7831479, 0.       ]],
              dtype=float32)

        """

        if self.neighborsStruct is None :
            self.neighborsStruct = neighbor_list('dij', self.atoms, cutoff)
            #neighborsStruct =(distanceVectore, index1, index2)

        dm = np.zeros( (self.nAt, self.nAt), dtype=np.float32 )
        for (dist,idx1,idx2) in zip(*self.neighborsStruct):
            dm[idx1,idx2] = dist

        return dm


    def neighborsOld(self, cutoff):
        """ 
        compute the neighbor distances up to the cutoff

        This returns 3 arrays (distVec, index1, index2)
          len(index1)=len(index2) = number of neighbors
          distVec[index1,index2] is the vector with distances

        >>> mol = GDBMol(demoMol['C'])
        >>> mol.neighbors(3)
        (array([1.09195306, 1.09195162, 1.09194638, 1.09194754, 1.09195306,
               1.78311976, 1.7831475 , 1.78315669, 1.78314839, 1.78315766,
               1.78311976, 1.09195162, 1.09194638, 1.7831475 , 1.78315766,
               1.78314787, 1.09194754, 1.78315669, 1.78314839, 1.78314787]), 
         array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]), array([1, 2, 3, 4, 0, 2, 3, 4, 4, 3, 1, 0, 0, 1, 2, 4, 0, 1, 2, 3]))
        """

        if self.neighborsStruct is None :
            self.neighborsStruct = neighbor_list('dij', self.atoms, cutoff)
        return self.neighborsStruct


    @property
    def muOld(self):
        return self.properties['mu']


demoMols = {}
demoMols['C'] = re.sub("  +", "\t", string.strip_spaces_as_in_first_line('''
    5
    gdb 1    157.7118    157.70997    157.70699    0.    13.21    -0.3877    0.1171    0.5048    35.3641    0.044749    -40.47893    -40.476062    -40.475117    -40.498597    6.469    
    C    -0.0126981359     1.0858041578     0.0080009958    -0.535689
    H     0.002150416     -0.0060313176     0.0019761204     0.133921
    H     1.0117308433     1.4637511618     0.0002765748     0.133922
    H    -0.540815069      1.4475266138    -0.8766437152     0.133923
    H    -0.5238136345     1.4379326443     0.9063972942     0.133923
    1341.307    1341.3284    1341.365    1562.6731    1562.7453    3038.3205    3151.6034    3151.6788    3151.7078
    C    C    
    InChI=1S/CH4/h1H4    InChI=1S/CH4/h1H4
    '''))

demoMols['C#C'] = re.sub("  +", "\t", string.strip_spaces_as_in_first_line('''
    4
    mist    157.7118    157.70997    157.70699    0.    13.21    -0.3877    0.1171    0.5048    35.3641    0.044749    -40.47893    -40.476062    -40.475117    -40.498597    6.469    
    C     0.0              0.0              0.0             -0.535689
    C     2.0              0.0              0.0              0.133921
    H     3.0              0.0              0.0              0.133922
    H    -1.0              0.0              0.0              0.133923
    1341.307    1341.3284    1341.365    1562.6731    1562.7453    3038.3205    3151.6034    3151.6788    3151.7078
    C#C  C#C  
    InChI=1S/CH4/h1H4    InChI=1S/CH4/h1H4
    '''))

demoMols['CC#C'] = re.sub("  +", "\t", string.strip_spaces_as_in_first_line('''
    7
    gdb 9    160.28041    8.59323    8.59321    0.7156    28.78    -0.2609    0.0613    0.3222    177.1963    0.05541    -116.609549    -116.60555    -116.604606    -116.633775    12.482    
    C    -0.0178210241     1.4643578827     0.0100939689    -0.493017
    C     0.0020881587     0.0095077743     0.0020119979     0.286586
    C     0.018340851    -1.1918051787    -0.0045050842    -0.443025
    H     0.9978221     1.8742534891     0.0026060629     0.151087
    H    -0.5422043423     1.8580117813    -0.867211921     0.151092
    H    -0.5253330614     1.8483435616     0.9014813825     0.151097
    H     0.0323165781    -2.2531482303    -0.0102599871     0.196182
    340.0245    340.055    619.3467    619.3538    946.5116    1055.9131    1055.9288    1414.1864    1477.8966    1477.9352    2245.742    3031.4807    3093.8747    3093.9562    3510.1131
    CC#C    CC#C    
    InChI=1S/C3H4/c1-3-2/h1H,2H3    InChI=1S/C3H4/c1-3-2/h1H,2H3
    '''))

demoMols['O'] = re.sub("  +", "\t", string.strip_spaces_as_in_first_line('''
    3
    mist    157.7118    157.70997    157.70699    0.    13.21    -0.3877    0.1171    0.5048    35.3641    0.044749    -40.47893    -40.476062    -40.475117    -40.498597    6.469    
    O     0.0              0.0              0.0             -0.535689
    H     1.0              0.0              0.0              0.133921
    H     0.0              1.0              0.0              0.133922
    1341.307    1341.3284    1341.365    1562.6731    1562.7453    3038.3205    3151.6034    3151.6788    3151.7078
    O  O  
    InChI=1S/CH4/h1H4    InChI=1S/CH4/h1H4
    '''))



demoMols['angleTest'] = re.sub("  +", "\t", string.strip_spaces_as_in_first_line('''
    4
    mist    157.7118    157.70997    157.70699    0.    13.21    -0.3877    0.1171    0.5048    35.3641    0.044749    -40.47893    -40.476062    -40.475117    -40.498597    6.469    
    O     0.0              0.0              0.0             -0.535689
    H     1.0              0.0              0.0              0.133921
    H     0.0              2.0              0.0              0.133922
    H    -3.0              0.0              0.0              0.133921
    1341.307    1341.3284    1341.365    1562.6731    1562.7453    3038.3205    3151.6034    3151.6788    3151.7078
    O  O  
    InChI=1S/CH4/h1H4    InChI=1S/CH4/h1H4
    '''))

demoMols['angleTest2.1'] = re.sub("  +", "\t", string.strip_spaces_as_in_first_line('''
    3
    mist    157.7118    157.70997    157.70699    0.    13.21    -0.3877    0.1171    0.5048    35.3641    0.044749    -40.47893    -40.476062    -40.475117    -40.498597    6.469    
    H     0.0              0.0              0.0             -0.535689
    N     1.0              0.0              0.0              0.133921
    O     0.0              2.0              0.0              0.133922
    1341.307    1341.3284    1341.365    1562.6731    1562.7453    3038.3205    3151.6034    3151.6788    3151.7078
    O  O  
    InChI=1S/CH4/h1H4    InChI=1S/CH4/h1H4
    '''))
demoMols['angleTest2.2'] = re.sub("  +", "\t", string.strip_spaces_as_in_first_line('''
    3
    mist    157.7118    157.70997    157.70699    0.    13.21    -0.3877    0.1171    0.5048    35.3641    0.044749    -40.47893    -40.476062    -40.475117    -40.498597    6.469    
    H     0.0              0.0              0.0             -0.535689
    O     0.0              2.0              0.0              0.133922
    N     1.0              0.0              0.0              0.133921
    1341.307    1341.3284    1341.365    1562.6731    1562.7453    3038.3205    3151.6034    3151.6788    3151.7078
    O  O  
    InChI=1S/CH4/h1H4    InChI=1S/CH4/h1H4
    '''))

demoMols['angleTest3.1'] = re.sub("  +", "\t", string.strip_spaces_as_in_first_line('''
    3
    mist    157.7118    157.70997    157.70699    0.    13.21    -0.3877    0.1171    0.5048    35.3641    0.044749    -40.47893    -40.476062    -40.475117    -40.498597    6.469    
    O     0.0              0.0              0.0             -0.535689
    H     1.0              0.0              0.0              0.133921
    H     0.0              2.0              0.0              0.133922
    1341.307    1341.3284    1341.365    1562.6731    1562.7453    3038.3205    3151.6034    3151.6788    3151.7078
    O  O  
    InChI=1S/CH4/h1H4    InChI=1S/CH4/h1H4
    '''))
demoMols['angleTest3.2'] = re.sub("  +", "\t", string.strip_spaces_as_in_first_line('''
    3
    mist    157.7118    157.70997    157.70699    0.    13.21    -0.3877    0.1171    0.5048    35.3641    0.044749    -40.47893    -40.476062    -40.475117    -40.498597    6.469    
    O     0.0              0.0              0.0             -0.535689
    H     0.0              2.0              0.0              0.133922
    H     1.0              0.0              0.0              0.133921
    1341.307    1341.3284    1341.365    1562.6731    1562.7453    3038.3205    3151.6034    3151.6788    3151.7078
    O  O  
    InChI=1S/CH4/h1H4    InChI=1S/CH4/h1H4
    '''))

demoMols['bad'] = re.sub("  +", "\t", string.strip_spaces_as_in_first_line('''
    16
    gdb 15149    12.5895    0.61947    0.61732    0.3397    96.22    -0.2162    -0.0013    0.2149    1828.4545    0.129238    -309.418301    -309.409421    -309.408476    -309.453456    30.542    
    C    -0.003194447     1.485015887    -0.0013993727    -0.517617
    C     0.0067236133     0.0331280553    -0.0020682711     0.228098
    C     0.0148673445    -1.1777575856    -0.0023586769    -0.189969
    C     0.0243127377    -2.5367311501    -0.0025537062    -0.227372
    C     0.0255012483    -3.7493732482     0.0021990519     0.327519
    C     0.0408718859    -5.1781327068    -0.0035788818    -0.199911
    C    -0.4111801388    -5.9566355492     1.2256690037    -0.250773
    C    -1.2663277285    -5.9609304477    -0.0042278496    -0.250743
    H     1.0157618112     1.8888035358    -0.0120605803     0.154186
    H    -0.5268500894     1.8815537535    -0.8788864916     0.153377
    H    -0.5052265355     1.877695768     0.890361409     0.153909
    H     0.8550754987    -5.6243667128    -0.5680673918     0.115385
    H     0.1487090279    -6.8511016238     1.4740814323     0.121064
    H    -0.7446856334    -5.3767131661     2.0776524549     0.130875
    H    -2.1830046628    -5.3839259836     0.008994718     0.130887
    H    -1.3013256219    -6.8583992958    -0.6113357779     0.121085
    23.4312    69.6142    75.568    173.687    186.0085    320.4095    322.9143    436.1148    486.4382    545.5171    580.4737    594.9207    783.8753    819.2133    838.0479    886.392    967.2224    1047.826    1048.3781    1061.1884    1079.7808    1108.5702    1113.9051    1201.0782    1207.2053    1288.2733    1399.3397    1417.0735    1466.5045    1472.3465    1473.5207    1495.4518    2275.1272    2378.4287    3021.4721    3082.1274    3082.7741    3139.9758    3141.8391    3147.0311    3227.3271    3242.4012
    CC#CC#CC1CC1    CC#CC#CC1CC1    
    InChI=1S/C8H8/c1-2-3-4-5-8-6-7-8/h8H,6-7H2,1H3    InChI=1S/C8H8/c1-2-3-4-5-8-6-7-8/h8H,6-7H2,1H3
    '''))


demoMols['H2O_HF'] = re.sub("  +", "\t", string.strip_spaces_as_in_first_line('''
    3
    mist    157.7118    157.70997    157.70699    0.    13.21    -0.3877    0.1171    0.5048    35.3641    0.044749    -76.0533051650    -76.0533051650    -40.475117    -40.498597    6.469    
    O    -0.278918        -0.206566         0.145968         0
    H    -0.022956         0.661093         0.405987         0
    H     0.301873        -0.454627        -0.551956         0
    1341.307    1341.3284    1341.365    1562.6731    1562.7453    3038.3205    3151.6034    3151.6788    3151.7078
    O  O  
    InChI=1S/CH4/h1H4    InChI=1S/CH4/h1H4
    '''))
