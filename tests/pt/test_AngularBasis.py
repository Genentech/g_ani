
import pprint
import re

import numpy as np
import numpy.testing as npt
import torch

import cdd_chem.util.string as string
from cdd_chem.util.io import warn  # noqa: F401; # pylint: disable=W0611
from ml_qm.ANIMol import ANIMol
from ml_qm.GDBMol import GDBMol
from ml_qm.GDBMol import demoMols
from ml_qm.pt.AngularBasis import GaussianCosineBasis, GaussianTANIAngularBasis, \
    BumpAngularBasis, GaussianAngularBasis, Gaussian2AngularBasis, \
    Bump2AngularBasis, Bump3AngularBasis
from ml_qm.pt.PTMol import PTMol
from ml_qm.pt.ThreeDistBasis import GaussianThirdDist2Basis, \
    GaussianThirdDistCombBasis, GaussianThirdDistBasis, Bump3ThirdDistanceBasis
from ml_qm.pt.ThreeDistSqrBasis import GaussianThirdDistSqrBasis

pp = pprint.PrettyPrinter(indent=2, width=2000)

torch.set_printoptions(linewidth=200)
torch.set_printoptions(precision=2, threshold=9999, linewidth=9999, sci_mode=False)

_molRadTest = re.sub("  +", "\t", string.strip_spaces_as_in_first_line('''
    4
    mist    157.7118    157.70997    157.70699    0.    13.21    -0.3877    0.1171    0.5048    35.3641    0.044749    -40.47893    -40.476062    -40.475117    -40.498597    6.469    
    O     0.0              0.0              0.0             -0.535689
    H     1.0              0.0              0.0              0.133921
    H     2.0              0.0              0.0              0.133922
    H     3.0              0.0              0.0              0.133921
    1341.307    1341.3284    1341.365    1562.6731    1562.7453    3038.3205    3151.6034    3151.6788    3151.7078
    O  O  
    InChI=1S/CH4/h1H4    InChI=1S/CH4/h1H4
    '''))




def test_GaussThirdDistSqrDesc():
    """
    Model compounds atom indexes are same as x/y coordinates:
            0,0,0, 1,0,0, 0,2,0, -3,0,0
    diagonal distances are 1-2: 2.3,    2-3: 3.6

                  2
                  |
                  |
    -3 ---------- 0 -----1
    """

    #                         atomTypes, nAngles, nRadial, rMin, rMax, aCent,rCent, halfWidth, angleHalfWidth, cutoff
    ab = GaussianThirdDistSqrBasis([1,8],   3,       3,       1,    3,     None, None, 1,         0.125,         9)
    mol   = PTMol(GDBMol(demoMols['angleTest']))

    # rShift     1       2       3        1       2      3       1       2      3
    # aShift     0       0       0        .5      .5     .5      1       1      1         Third dist
    expectO0 = [ 0.   , 0.   , 0.   , 0.429, 0.759, 0.333, 0.045, 0.727, 0.045,   # O(-H)-H
                 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,   # O(-H)-O
                 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.     ] # O(-O)-O
    expectH1 = [ 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,   # H(-H)-H
                 0.001, 0.285, 0.285, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,   # H(-H)-O
                 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.    ]  # H(-O)-O

    coords = np.asarray([mol.baseMol.xyz,mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums,mol.atNums], dtype=np.int64)

    res = ab.computeDescriptorBatch(coords, atTypes)

    npt.assert_array_almost_equal(expectO0, res[0,0], 3)  # o
    npt.assert_array_almost_equal(expectH1, res[0,1], 3)  # H


    ab = GaussianThirdDist2Basis([1,8],   3,       3,       1,    3,     None, None, 1,         0.125,         9)
    # rShift     1       2       3        1       2      3       1       2      3
    # aShift     0       0       0        .5      .5     .5      1       1      1         Third dist
    expectO0 = [ 0.    , 0.    , 0.    , 0.0388, 0.0444, 0.0083, 0.3637, 0.0455, 0.3637,   # O(-H)-H
                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,   # O(-H)-O
                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.      ] # O(-O)-O
    expectH1 = [ 0.    , 0.    , 0.    , 0.0009, 0.0541, 0.0165, 0.    , 0.    , 0.    ,   # H(-H)-H
                 0.2846, 0.0178, 0.0178, 0.0355, 0.0322, 0.0069, 0.    , 0.    , 0.    ,   # H(-H)-O
                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.     ]  # H(-O)-O
    res = ab.computeDescriptorBatch(coords, atTypes)

    npt.assert_array_almost_equal(expectO0, res[0,0], 3)  # o
    npt.assert_array_almost_equal(expectH1, res[0,1], 3)  # H


def test_GaussThirdDistSqrDesc2():
    """
    Model compounds atom indexes are same as x/y coordinates:
            0,0,0, 1,0,0, 0,2,0,
            0,0,0, 0,2,0, 1,0,0,

         O2          O1
          |           |
          |           |
         H0 - 1N     H0 - 2N
    """

    #                              atomTypes, nAngles, nRadial, rMin, rMax, aCent,rCent, halfWidth, angleHalfWidth, cutoff
    ab = GaussianThirdDistSqrBasis([1,7,8],   3,       3,       1,    3,     None, None, 1,         0.250,         9)
    mol1   = PTMol(GDBMol(demoMols['angleTest2.1']))
    mol2   = PTMol(GDBMol(demoMols['angleTest2.2']))

    # I(-J)K            X(-H)H                             X(-N)H                                    X(O)H                                 X(N)N                            X(N)O                               X(O)O                  X
    # rShift     1  2  3  1  2  3  1  2  3      1     2     3     1  2  3  1  2  3       1     2     3  1     2     3  1  2  3      1  2  3  1  2  3  1  2  3     1  2  3  1     2     3  1  2  3      1  2  3  1  2  3  1  2  3
    # aShift     0  0  0  .5 .5 .5 1  1  1      0     0     0     .5 .5 .5 1  1  1       0     0     0  .5    .5    .5 1  1  1      0  0  0  .5 .5 .5 1  1  1     0  0  0  .5    .5    .5 1  1  1      0  0  0  .5 .5 .5 1  1  1
    expect = [[[ 0, 0, 0, 0, 0, 0, 0, 0, 0,     0.  , 0.  , 0.  , 0, 0, 0, 0, 0, 0,      0.  , 0.  , 0, 0.  , 0.  , 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0,    0, 0, 0, 0.43, 0.43, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0.],  # H0
               [ 0, 0, 0, 0, 0, 0, 0, 0, 0,     0.  , 0.  , 0.  , 0, 0, 0, 0, 0, 0,      0.01, 0.02, 0, 0.03, 0.06, 0, 0, 0,
                   0,     0, 0, 0, 0, 0, 0, 0, 0, 0,    0, 0, 0, 0.  , 0.  , 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0.],  # N1
               [ 0, 0, 0, 0, 0, 0, 0, 0, 0,     0.02, 0.64, 0.08, 0, 0, 0, 0, 0, 0,      0.  , 0.  , 0, 0.  , 0.  , 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0,    0, 0, 0, 0.  , 0.  , 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0.]], # O2
              [[ 0, 0, 0, 0, 0, 0, 0, 0, 0,     0.  , 0.  , 0.  , 0, 0, 0, 0, 0, 0,      0.  , 0.  , 0, 0.  , 0.  , 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0,    0, 0, 0, 0.43, 0.43, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0.],  # H4
               [ 0, 0, 0, 0, 0, 0, 0, 0, 0,     0.02, 0.64, 0.08, 0, 0, 0, 0, 0, 0,      0.  , 0.  , 0, 0.  , 0.  , 0, 0, 0,
                   0,     0, 0, 0, 0, 0, 0, 0, 0, 0,    0, 0, 0, 0.  , 0.  , 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0.],  # O5
               [ 0, 0, 0, 0, 0, 0, 0, 0, 0,     0.  , 0.  , 0.  , 0, 0, 0, 0, 0, 0,      0.01, 0.02, 0, 0.03, 0.06, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0,    0, 0, 0, 0.  , 0.  , 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0.]]] # N6

    coords = np.asarray([mol1.baseMol.xyz,mol2.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol1.atNums,mol2.atNums], dtype=np.int64)

    res = ab.computeDescriptorBatch(coords, atTypes)

    npt.assert_array_almost_equal(expect, res, 2)



def test_GaussThirdDistSqrDesc3():
    """
    Model compounds atom indexes are same as x/y coordinates:
            0,0,0, 1,0,0, 0,2,0,
            0,0,0, 0,2,0, 1,0,0,

         H2          H1
          |           |
          |           |
         O0 - 1H     O0 - 2H
    """

    #                              atomTypes, nAngles, nRadial, rMin, rMax, aCent,rCent, halfWidth, angleHalfWidth, cutoff
    ab = GaussianThirdDistSqrBasis([1,8],   3,       3,       1,    3,     None, None, 1,         0.250,         9)
    mol1   = PTMol(GDBMol(demoMols['angleTest3.1']))
    mol2   = PTMol(GDBMol(demoMols['angleTest3.2']))

    # I(-J)K            X(-H)H                             X(-O)H                                    X(O)O                    X
    # rShift     1  2  3  1     2     3  1  2  3      1     2     3     1    2      3  1  2  3      1  2  3  1  2  3  1  2  3
    # aShift     0  0  0  .5    .5    .5 1  1  1      0     0     0     .5   .5     .5 1  1  1      0  0  0  .5 .5 .5 1  1  1
    expect = [[[ 0, 0, 0, 0.43, 0.43, 0, 0, 0, 0,     0,    0,    0,    0,    0,    0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0],   # O0
               [ 0, 0, 0, 0,    0,    0, 0, 0, 0,     0.01, 0.02, 0,    0.03, 0.06, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0],   # H1
               [ 0, 0, 0, 0,    0,    0, 0, 0, 0,     0.02, 0.64, 0.08, 0,    0,    0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0]],  # H2
              [[ 0, 0, 0, 0.43, 0.43, 0, 0, 0, 0,     0,    0,    0,    0,    0,    0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0],   # O0
               [ 0, 0, 0, 0,    0,    0, 0, 0, 0,     0.02, 0.64, 0.08, 0,    0,    0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0],   # H1
               [ 0, 0, 0, 0,    0,    0, 0, 0, 0,     0.01, 0.02, 0,    0.03, 0.06, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0]]]  # H2

    coords = np.asarray([mol1.baseMol.xyz,mol2.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol1.atNums,mol2.atNums], dtype=np.int64)

    res = ab.computeDescriptorBatch(coords, atTypes)

    npt.assert_array_almost_equal(expect, res, 2)


def test_GaussThirdDistCombDesc():
    """
    Model compounds atom idnexes are same as x/y coordinates:
            0,0,0, 1,0,0, 0,2,0, -3,0,0
    diagonal distances are 1-2: 2.3,    2-3: 3.6

                  2
                  |
                  |
    -3 ---------- 0 -----1
    """

    #                               atomTypes,n3rd, nRadial, rMin, rMax, aCent,rCent, halfWidth, 3rdFact, cutoff
    ab = GaussianThirdDistCombBasis([1,8],      3,        3,    1,    3,  None, None,      1,    2,          9)
    mol   = PTMol(GDBMol(demoMols['angleTest']))

    # rShift     1       2       3        1       2      3       1       2      3
    # aShift     0       0       0        .5      .5     .5      1       1      1         Third dist
    expectO0 =  [0.0635, 0.1031, 0.0491, 0.2329, 0.4476, 0.184 , 0.165 , 0.9332, 0.1503,   # O(-H)-H
                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,   # O(-H)-O
                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ]   # O(-O)-O
#     expectO0 = [ 0.    , 0.    , 0.    , 0.2085, 0.258 , 0.0526, 0.1151, 0.011 , 0.1142,   # O(-H)-H
#                  0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,   # O(-H)-O
#                  0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.      ] # O(-O)-O
    expectH1 =  [0.0011, 0.0152, 0.0966, 0.0014, 0.0222, 0.3118, 0.0009, 0.0105, 0.0489,   # H(-H)-H
                 0.0941, 0.2748, 0.1558, 0.1463, 0.3127, 0.0691, 0.0445, 0.0678, 0.0178,   # H(-H)-O
                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ]   # H(-O)-O
#     expectH1 = [ 0.    , 0.    , 0.    , 0.    , 0.0058, 0.001 , 0.    , 0.    , 0.    ,   # H(-H)-H
#                  0.0173, 0.0037, 0.0007, 0.1768, 0.1556, 0.0254, 0.    , 0.    , 0.    ,   # H(-H)-O
#                  0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.     ]  # H(-O)-O


    coords = np.asarray([mol.baseMol.xyz,mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums,mol.atNums], dtype=np.int64)

    res = ab.computeDescriptorBatch(coords, atTypes)

    npt.assert_array_almost_equal(expectO0, res[0,0], 3)  # o
    npt.assert_array_almost_equal(expectH1, res[0,1], 3)  # H



def test_GaussianCosineBasis():
    """
    Model compounds atom idnexes are same as x/y coordinates:
            0,0,0, 1,0,0, 0,2,0, -3,0,0
    diagonal distances are 1-2: 2.3,    2-3: 3.6

                  2
                  |
                  |
    -3 ---------- 0 -----1
    """

    #                      atomTypes, nAngles, nRadial, rMin, rMax, aCent,rCent, halfWidth, angleHalfWidth, cutoff
    ab = GaussianCosineBasis([1,8],   3,       3,       1,    3,     None, None, 1,         0.25,         9)
    mol   = PTMol(GDBMol(demoMols['angleTest']))

    #np.set_printoptions(4,linewidth=90,suppress=True)
    # res.numpy(
    # rShift     1       2       3        1       2      3       1       2      3
    # aShift    -1      -1      -1        0       0      0       1       1      1         Third dist
    expectO0 = [ 0.    , 0.    , 0.    , 0.4295, 0.7593, 0.3328, 0.0455, 0.7274, 0.0455,   # O(-H)-H
                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,   # O(-H)-O
                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ]   # O(-O)-O
    expectH1 = [ 0.    , 0.    , 0.    , 0.    , 0.    , 0.0001, 0.    , 0.    , 0.    ,   # H(-H)-H
                 0.0011, 0.2846, 0.2846, 0.    , 0.0001, 0.    , 0.    , 0.    , 0.    ,   # H(-H)-O
                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ]   # H(-O)-O

    coords = np.asarray([mol.baseMol.xyz,mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums,mol.atNums], dtype=np.int64)

    res = ab.computeDescriptorBatch(coords, atTypes)

    npt.assert_array_almost_equal(expectO0, res[0,0], 3)  # o
    npt.assert_array_almost_equal(expectH1, res[0,1], 3)  # H





def test_GaussThirdDistDesc():
    """
    Model compounds atom idnexes are same as x/y coordinates:
            0,0,0, 1,0,0, 0,2,0, -3,0,0
    diagonal distances are 1-2: 2.3,    2-3: 3.6

                  2
                  |
                  |
    -3 ---------- 0 -----1
    """

    #                         atomTypes, nAngles, nRadial, rMin, rMax, aCent,rCent, halfWidth, angleHalfWidth, cutoff
    ab = GaussianThirdDistBasis([1,8],   3,       3,       1,    3,     None, None, 1,         0.125,         9)
    mol   = PTMol(GDBMol(demoMols['angleTest']))

    # rShift     1       2       3        1       2      3       1       2      3
    # aShift     0       0       0        .5      .5     .5      1       1      1         Third dist
    expectO0 = [ 0.    , 0.    , 0.    , 0.0362, 0.0418, 0.0058, 0.0455, 0.7274, 0.0455,   # O(-H)-H
                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,   # O(-H)-O
                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.      ] # O(-O)-O
    expectH1 = [ 0.    , 0.    , 0.    , 0.    , 0.0039, 0.1214, 0.    , 0.    , 0.    ,   # H(-H)-H
                 0.0011, 0.2846, 0.2846, 0.0243, 0.0467, 0.0004, 0.    , 0.    , 0.    ,   # H(-H)-O
                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.     ]  # H(-O)-O

    coords = np.asarray([mol.baseMol.xyz,mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums,mol.atNums], dtype=np.int64)

    res = ab.computeDescriptorBatch(coords, atTypes)

    npt.assert_array_almost_equal(expectO0, res[0,0], 3)  # o
    npt.assert_array_almost_equal(expectH1, res[0,1], 3)  # H


    ab = GaussianThirdDist2Basis([1,8],   3,       3,       1,    3,     None, None, 1,         0.125,         9)
    # rShift     1       2       3        1       2      3       1       2      3
    # aShift     0       0       0        .5      .5     .5      1       1      1         Third dist
    expectO0 = [ 0.    , 0.    , 0.    , 0.0388, 0.0444, 0.0083, 0.3637, 0.0455, 0.3637,   # O(-H)-H
                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,   # O(-H)-O
                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.      ] # O(-O)-O
    expectH1 = [ 0.    , 0.    , 0.    , 0.0009, 0.0541, 0.0165, 0.    , 0.    , 0.    ,   # H(-H)-H
                 0.2846, 0.0178, 0.0178, 0.0355, 0.0322, 0.0069, 0.    , 0.    , 0.    ,   # H(-H)-O
                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.     ]  # H(-O)-O
    res = ab.computeDescriptorBatch(coords, atTypes)

    npt.assert_array_almost_equal(expectO0, res[0,0], 3)  # o
    npt.assert_array_almost_equal(expectH1, res[0,1], 3)  # H


def test_BumpThirdDistDesc():
    """
    Model compounds atom idnexes are same as x/y coordinates:
            0,0,0, 1,0,0, 0,2,0, -3,0,0
    diagonal distances are 1-2: 2.3,    2-3: 3.6

                  2
                  |
                  |
    -3 ---------- 0 -----1
    """

    #                         atomTypes, nAngles, nRadial, rMin, rMax, aCent,rCent, halfWidth, maxHWMult, 3rdDHalfWidth, max3rdDHWMult, cutoff
    ab = Bump3ThirdDistanceBasis([1,8],   3,       3,       1,    3,     None, None, 1,        3,          0.125,        3,             9)
    mol   = PTMol(GDBMol(demoMols['angleTest']))

    # rShift     1       2       3        1       2      3       1       2      3
    # aShift     0       0       0        .5      .5     .5      1       1      1         Third dist
    expectO0 = [ 0.    , 0.    , 0.    , 0.2085, 0.258 , 0.0526, 0.1151, 0.011 , 0.1142,   # O(-H)-H
                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,   # O(-H)-O
                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.      ] # O(-O)-O
    expectH1 = [ 0.    , 0.    , 0.    , 0.    , 0.0058, 0.001 , 0.    , 0.    , 0.    ,   # H(-H)-H
                 0.0173, 0.0037, 0.0007, 0.1768, 0.1556, 0.0254, 0.    , 0.    , 0.    ,   # H(-H)-O
                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.     ]  # H(-O)-O


    coords = np.asarray([mol.baseMol.xyz,mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums,mol.atNums], dtype=np.int64)

    res = ab.computeDescriptorBatch(coords, atTypes)

    npt.assert_array_almost_equal(expectO0, res[0,0], 3)  # o
    npt.assert_array_almost_equal(expectH1, res[0,1], 3)  # H



def test_TorchAni():
    atomTypes= torch.LongTensor([1,6,7,8])
    xyz = torch.tensor(
        [[0.,0,0],
         [1,0,0],
         [0,2,0],
         [-2,0,0]])

    mol = PTMol(ANIMol("demo", 0.0, atomTypes, xyz))
    aCent = [0.1963, 0.5890, 0.9817, 1.3744, 1.7671, 2.1598, 2.5525, 2.9452]
    #                             atomTypes, nAngles, nRadial, rMin, rMax, aCent,rCent, halfWidth, angleHalfWidth, cutoff
    ab = GaussianTANIAngularBasis([1,6,7,8], None,       4,    0.9,  2.85, aCent, None, 0.5887050, 0.187052556,       3.5)

    coords = np.asarray([mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums], dtype=np.int64)
    res = ab.computeDescriptorBatch(coords, atTypes)
    np.set_printoptions(4,linewidth=268,suppress=True, threshold=10000)
    r,_=res[0].sort(descending=True)
    r = r[:,0:23].numpy()

    expected = \
      [[0.4543, 0.4542, 0.4541, 0.1612, 0.1611, 0.0439, 0.0439, 0.037 , 0.037 , 0.037 , 0.026 , 0.026 , 0.026 , 0.0131, 0.0131, 0.0092, 0.0092, 0.0092, 0.0036, 0.0036, 0.0021, 0.0021, 0.0021],
       [0.3982, 0.2546, 0.0515, 0.0429, 0.0275, 0.0176, 0.0164, 0.013 , 0.0117, 0.0105, 0.0067, 0.0062, 0.0043, 0.004 , 0.0036, 0.0035, 0.0021, 0.001 , 0.0009, 0.0009, 0.0008, 0.0005, 0.0005],
       [0.1876, 0.1199, 0.0349, 0.0348, 0.0243, 0.02  , 0.0186, 0.015 , 0.0128, 0.0119, 0.011 , 0.011 , 0.0096, 0.0028, 0.0028, 0.0027, 0.0026, 0.0024, 0.0019, 0.0017, 0.0009, 0.0009, 0.0007],
       [0.0349, 0.0348, 0.0138, 0.011 , 0.011 , 0.0106, 0.0062, 0.0062, 0.0028, 0.0028, 0.0011, 0.0009, 0.0009, 0.0009, 0.0005, 0.0005, 0.0001, 0.0001, 0.0001, 0.0001, 0.    , 0.    , 0.    ]]

    npt.assert_allclose(r, expected, atol=1e-2)


    atomTypes= torch.LongTensor([ 6,1,1,1,1])
    xyz = torch.tensor(
        [[-0.0035,  0.0102,  0.0194],
         [-0.7955,  0.5767, -0.5472],
         [-0.3938, -0.9799,  0.2723],
         [ 0.6345,  0.4474,  0.9357],
         [ 0.5958, -0.1652, -0.8916]])

    mol = PTMol(ANIMol("CH4", 0.0, atomTypes, xyz))

    coords = np.asarray([mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums], dtype=np.int64)
    res = ab.computeDescriptorBatch(coords, atTypes)
    r,_=res[0].sort(descending=True)
    r = r[:,0:23].numpy()

    expected = \
      [[3.4927, 2.5168, 1.2364, 1.0453, 0.8851, 0.3294, 0.2859, 0.1593, 0.0348, 0.0111, 0.0061, 0.0031, 0.0005, 0.0005, 0.0001, 0.0001, 0.0001, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
       [1.7264, 0.7443, 0.7137, 0.4903, 0.4008, 0.3321, 0.185 , 0.171 , 0.1301, 0.1166, 0.0816, 0.0401, 0.0353, 0.0316, 0.0172, 0.0123, 0.0099, 0.0068, 0.0059, 0.0031, 0.0018, 0.0015, 0.0013],
       [1.9693, 0.8279, 0.7047, 0.3974, 0.3935, 0.3249, 0.2175, 0.1999, 0.1556, 0.0859, 0.0597, 0.0391, 0.0273, 0.0248, 0.0245, 0.0149, 0.0102, 0.005 , 0.0035, 0.0029, 0.0019, 0.0012, 0.0009],
       [1.7074, 0.7091, 0.5713, 0.3732, 0.2791, 0.2622, 0.1159, 0.1089, 0.0792, 0.0537, 0.0493, 0.0375, 0.0184, 0.0154, 0.0123, 0.0117, 0.0065, 0.0045, 0.0017, 0.0017, 0.0009, 0.0009, 0.0003],
       [1.797 , 0.9617, 0.8222, 0.706 , 0.3306, 0.2489, 0.2425, 0.2007, 0.163 , 0.0891, 0.0565, 0.0496, 0.0352, 0.0303, 0.0234, 0.0161, 0.0101, 0.0097, 0.0052, 0.005 , 0.0042, 0.0007, 0.0005]]
    npt.assert_allclose(r, expected, atol=1e-4)


def test_C3H7N():
    """ test compound that differed between batch and single descriptor computation"""
    energy    = -4.514764
    atomTypes = torch.LongTensor([ 7,  6,  6,  6,  1,  1,  1,  1,  1,  1,  1])
    xyz = torch.tensor(
        [[ 1.8022, -0.3364, -0.1414],
         [ 0.5561,  0.4222,  0.1359],
         [-0.6964, -0.3107,  0.2755],
         [-1.9377,  0.0966, -0.1634],
         [ 1.8432, -0.4087, -1.1873],
         [ 2.6531,  0.2257,  0.2052],
         [ 0.5977,  0.9778,  1.1304],
         [ 0.2634,  1.3216, -0.5458],
         [-0.6373, -1.2803,  0.7489],
         [-2.2364,  0.9335, -0.7998],
         [-2.7815, -0.5174,  0.0396]])

    mol = PTMol(ANIMol("C3H7N", energy, atomTypes, xyz))
    #                     atomTypes, nAngles, nRadial,   rMin, rMax, aCent,rCent, halfWidth, maxWidthMultiplier, angleHalfWidth, maxAngleWidthMultiplier
    ab = BumpAngularBasis([1,6,7,8],        2,      2,      1.5, 3,  None, None,   1.5,      3,                   1.57,                3)
    sdescs = ab.computeDescriptors(mol)
    descBatch = ab.computeDescriptorBatch(np.expand_dims(xyz.numpy(),0), atomTypes.numpy())

    countDiffs = 0
    for atT, bd in zip(atomTypes, descBatch[0]):
        sdesc = sdescs[atT.item()]
        minDiff = 999999
        minPair = (-1,-1)
        for sd in sdesc:
            sd = sd.reshape(-1)
            diff = (sd - bd).abs().sum()
            if diff < minDiff:
                minPair = (sd,bd)
                minDiff = diff
        if minDiff > 0.01:
            warn("AtomType = %s" % atT)
            warn("diff %s\nsingle %s\nbatch  %s\n" % (minPair[0] - minPair[1],  minPair[0],minPair[1]))
            countDiffs += 1

    #singleADesc = ab._computeAngularDescriptors(mol)
    #for d in singleADesc[0]: warn(d)

    assert countDiffs == 0



def test_GaussAngularDesc():
    """
    Model compounds atom idnexes are same as x/y coordinates:
            0,0,0, 1,0,0, 0,2,0, -3,0,0
    diagonal distances are 1-2: 2.3,    2-3: 3.6

                  2
                  |
                  |
    -3 ---------- 0 -----1
    """

    #                         atomTypes, nAngles, nRadial, rMin, rMax, aCent,rCent, halfWidth, angleHalfWidth, cutoff
    ab = GaussianAngularBasis([1,8],     3,       3,       1,    3,     None, None, 1,         0.4286,         9)
    mol   = PTMol(GDBMol(demoMols['angleTest']))
    desc = ab.computeDescriptors(mol)

    # rShift      1        2        3         1        2       3        1        2       3
    # aShift      0        0        0        pi/2     pi/2     pi/2     pi       pi      pi       Angles
    expectH1 = [[ 0.0000,  0.0045,  0.1386,  0.0000,  0.0227,  0.6975,  0.0000,  0.0000,  0.0004], # H(-H)-H
                [ 0.0848,  0.7280,  0.5703,  0.4155,  0.8084,  0.0149,  0.0003,  0.0005,  0.0000], # H(-H)-O
                [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]] # H(-O)-O
    expectO0 = [[ 0.0134,  0.0237,  0.0104,  0.8604,  1.5414,  0.6670,  0.1043,  1.4785,  0.1013], # O(-H)-H
                [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000], # O(-H)-O
                [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]] # O(-O)-O

    npt.assert_array_almost_equal(desc[1][0].numpy(), expectH1, 3)
    npt.assert_array_almost_equal(desc[8][0].numpy(), expectO0, 3)


    coords = np.asarray([mol.baseMol.xyz,mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums,mol.atNums], dtype=np.int64)

    res = ab.computeDescriptorBatch(coords, atTypes)

    npt.assert_array_almost_equal(desc[1][0].reshape(-1).numpy(), res[0][1], 3)  # H
    npt.assert_array_almost_equal(desc[8][0].reshape(-1).numpy(), res[0][0], 2)  # o
    npt.assert_array_almost_equal(desc[1][0].reshape(-1).numpy(), res[1][1], 3)  # H
    npt.assert_array_almost_equal(desc[8][0].reshape(-1).numpy(), res[1][0], 2)  # o



    # test filtering by cutoff
    #                         atomTypes, nAngles, nRadial, rMin, rMax, aCent,rCent, halfWidth, angleHalfWidth, cutoff
    ab2 = GaussianAngularBasis([1,8],     3,       3,       1,    3,    None, None, 1,         0.4286,         3)
    atTypes = np.asarray([mol.atNums,[8,1,1,1]], dtype=np.int64)

    # rShift      1        2        3         1        2       3        1        2       3
    # aShift      0        0        0        pi/2     pi/2     pi/2     pi       pi      pi       Angles
    expectH1 = [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,   # H(-H)-H    SHOULD BE 0 BECAUSE of CUTOFF
                 0.0113,  0.0218,  0.0002,  0.0570,  0.1096,  0.0008,  0.0000,  0.0001,  0.0000,   # H(-H)-O
                 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]   # H(-O)-O
    expectO0 = [ 0.0029,  0.0029,  0.0000,  0.1875,  0.1875,  0.0007,  0.0029,  0.0029,  0.0000,   # O(-H)-H
                 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,   # O(-H)-O
                 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]   # O(-O)-O


    res = ab2.computeDescriptorBatch(coords, atTypes)
    npt.assert_array_almost_equal(expectH1, res[1][1], 3)  # H
    npt.assert_array_almost_equal(expectO0, res[1][0], 3)  # O



    # test order independence, swap C0 of second mol with H1
    cPos = coords[1,0].copy()
    coords[1,0] = coords[1,1]
    coords[1,1] = cPos
    atTypes = np.asarray([mol.atNums,[1,8,1,1]], dtype=np.int64)

    res = ab.computeDescriptorBatch(coords, atTypes)
    npt.assert_array_almost_equal(desc[1][0].reshape(-1).numpy(), res[1][0], 3)  # H
    npt.assert_array_almost_equal(desc[8][0].reshape(-1).numpy(), res[1][1], 2)  # o


def test_Gauss2AngularDesc():
    """
    Model compounds atom idnexes are same as x/y coordinates:
            0,0,0, 1,0,0, 0,2,0, -3,0,0
    diagonal distances are 1-2: 2.3,    2-3: 3.6

                  2
                  |
                  |
    -3 ---------- 0 -----1
    """

    #                          atomTypes, nAngles, nRadial, rMin, rMax, aCent,rCent, halfWidth, angleHalfWidth, cutoff
    ab = Gaussian2AngularBasis([1,8],     3,       3,       1,    3,     None, None, 1,         0.4286,         9)
    mol   = PTMol(GDBMol(demoMols['angleTest']))

    # rShift      1       2       3        1       2      3        1       2      3
    # aShift      0       0       0       pi/2    pi/2    pi/2     pi      pi     pi     Angles
    expectH1 = [  0.0010, 0.0617, 0.0188, 0.0052, 0.3106, 0.0945, 0.0000, 0.0002, 0.0001,  # H(-H)-H
                  0.6899, 0.1450, 0.0592, 0.6166, 0.5513, 0.1194, 0.0004, 0.0003, 0.0001,  # H(-H)-O
                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000 ] # H(-O)-O
    expectO0 = [  0.0149, 0.0252, 0.0118, 0.9627, 1.6150, 0.7686, 0.7423, 0.1161, 0.7392,  # O(-H)-H
                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,  # O(-H)-O
                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000 ] # O(-O)-O


    # single conf
    coords = np.asarray([mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums], dtype=np.int64)
    res = ab.computeDescriptorBatch(coords, atTypes)

    npt.assert_array_almost_equal(expectH1, res[0][1], 3)  # H
    npt.assert_array_almost_equal(expectO0, res[0][0], 3)  # C


    # two confs
    coords = np.asarray([mol.baseMol.xyz,mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums,mol.atNums], dtype=np.int64)
    res = ab.computeDescriptorBatch(coords, atTypes)

    npt.assert_array_almost_equal(expectH1, res[0][1], 3)  # H
    npt.assert_array_almost_equal(expectO0, res[0][0], 3)  # C
    npt.assert_array_almost_equal(expectH1, res[1][1], 3)  # H
    npt.assert_array_almost_equal(expectO0, res[1][0], 3)  # C

    # test order independence, swap C0 of second mol with H1
    cPos = coords[1,0].copy()
    coords[1,0] = coords[1,1]
    coords[1,1] = cPos
    atTypes = np.asarray([mol.atNums,[1,8,1,1]], dtype=np.int64)

    res = ab.computeDescriptorBatch(coords, atTypes)
    npt.assert_array_almost_equal(expectH1, res[1][0], 3)  # H
    npt.assert_array_almost_equal(expectO0, res[1][1], 3)  # C


def test_BumpAngularDesc():
    """
    Model compounds atom idnexes are same as x/y coordinates:
            0,0,0, 1,0,0, 0,2,0, -3,0,0
    diagonal distances are 1-2: 2.3,    2-3: 3.6

                  2
                  |
                  |
    -3 ---------- 0 -----1
    """

    #                     atomTypes, nAngles, nRadial, rMin, rMax, aCent,rCent, halfWidth, maxWidthMultiplier, angleHalfWidth, maxAngleWidthMultiplier
    ab = BumpAngularBasis([1,8],     3,       3,       1,    3,     None, None, 1,         5,                  0.4286,         8)
    mol   = PTMol(GDBMol(demoMols['angleTest']))
    desc = ab.computeDescriptors(mol)

    # rShift      1        2        3         1        2       3        1        2       3
    # aShift      0        0        0        pi/2     pi/2     pi/2      pi       pi      pi     Angles
    expectH1 = [[ 0.0000,  0.0023,  0.1403,  0.0000,  0.0113,  0.6955,  0.0000, 0.0000,  0.0007], # H(-H)-H
                [ 0.0494,  0.5979,  0.5001,  0.2444,  0.4936,  0.0091,  0.0002, 0.0005,  0.0000], # H(-H)-O
                [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000]] # H(-O)-O
    expectO0 = [[ 0.0086,  0.0173,  0.0086,  0.5008,  1.0173,  0.5008,  0.0507, 1.0173,  0.0507], # O(-H)-H
                [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000], # O(-H)-O
                [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000]] # O(-O)-O

    npt.assert_array_almost_equal(desc[1][0].numpy(), expectH1, 3)
    npt.assert_array_almost_equal(desc[8][0].numpy(), expectO0, 3)



    ##### single molecule

    coords = np.asarray([mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums], dtype=np.int64)
    res = ab.computeDescriptorBatch(coords, atTypes)

    npt.assert_array_almost_equal(desc[1][0].reshape(-1), res[0][1], 3)  # H
    npt.assert_array_almost_equal(desc[8][0].reshape(-1), res[0][0], 3)  # C



    ### 2 mols for batch

    coords = np.asarray([mol.baseMol.xyz,mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums,mol.atNums], dtype=np.int64)
    res = ab.computeDescriptorBatch(coords, atTypes)

    npt.assert_array_almost_equal(desc[1][0].reshape(-1), res[0][1], 3)  # H
    npt.assert_array_almost_equal(desc[8][0].reshape(-1), res[0][0], 3)  # C
    npt.assert_array_almost_equal(desc[1][0].reshape(-1), res[1][1], 3)  # H
    npt.assert_array_almost_equal(desc[8][0].reshape(-1), res[1][0], 3)  # C

    # test order independence, swap C0 of second mol with H1
    cPos = coords[1,0].copy()
    coords[1,0] = coords[1,1]
    coords[1,1] = cPos
    atTypes = np.asarray([mol.atNums,[1,8,1,1]], dtype=np.int64)

    res = ab.computeDescriptorBatch(coords, atTypes)
    npt.assert_array_almost_equal(desc[1][0].reshape(-1), res[1][0], 3)  # H
    npt.assert_array_almost_equal(desc[8][0].reshape(-1), res[1][1], 3)  # C


def test_Bump2AngularDesc():
    """
    Model compounds atom idnexes are same as x/y coordinates:
            0,0,0, 1,0,0, 0,2,0, -3,0,0
    diagonal distances are 1-2: 2.3,    2-3: 3.6

                  2
                  |
                  |
    -3 ---------- 0 -----1
    """

    #                      atomTypes, nAngles, nRadial, rMin, rMax, aCent,rCent, halfWidth, maxWidthMultiplier, angleHalfWidth, maxAngleWidthMultiplier
    ab = Bump2AngularBasis([1,8],     3,       3,       1,    3,     None, None, 1,         5,                  0.4286,         8)
    mol   = PTMol(GDBMol(demoMols['angleTest']))
    desc = ab.computeDescriptors(mol)

    # rShift      1       2       3        1       2      3        1       2      3
    # aShift      0       0       0       pi/2    pi/2    pi/2     pi      pi     pi     Angles
    expectH1 = [[ 0.0000, 0.0000, 0.0127, 0.0000, 0.0000, 0.0628, 0.0000, 0.0000, 0.0001], # H(-H)-H
                [ 0.0099, 0.0277, 0.0000, 0.0489, 0.1374, 0.0000, 0.0000, 0.0001, 0.0000], # H(-H)-O
                [ 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]] # H(-O)-O
    expectO0 = [[ 0.0035, 0.0071, 0.0035, 0.2051, 0.4109, 0.2051, 0.0035, 0.0491, 0.0035], # O(-H)-H
                [ 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000], # O(-H)-O
                [ 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]] # O(-O)-O

    npt.assert_array_almost_equal(desc[1][0].numpy(), expectH1, 3)
    npt.assert_array_almost_equal(desc[8][0].numpy(), expectO0, 3)


    # single conf
    coords = np.asarray([mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums], dtype=np.int64)
    res = ab.computeDescriptorBatch(coords, atTypes)

    npt.assert_array_almost_equal(desc[1][0].reshape(-1).numpy(), res[0][1], 3)  # H
    npt.assert_array_almost_equal(desc[8][0].reshape(-1).numpy(), res[0][0], 3)  # C


    # two confs
    coords = np.asarray([mol.baseMol.xyz,mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums,mol.atNums], dtype=np.int64)
    res = ab.computeDescriptorBatch(coords, atTypes)

    npt.assert_array_almost_equal(desc[1][0].reshape(-1).numpy(), res[0][1], 3)  # H
    npt.assert_array_almost_equal(desc[8][0].reshape(-1).numpy(), res[0][0], 3)  # C
    npt.assert_array_almost_equal(desc[1][0].reshape(-1).numpy(), res[1][1], 3)  # H
    npt.assert_array_almost_equal(desc[8][0].reshape(-1).numpy(), res[1][0], 3)  # C

    # test order independence, swap C0 of second mol with H1
    cPos = coords[1,0].copy()
    coords[1,0] = coords[1,1]
    coords[1,1] = cPos
    atTypes = np.asarray([mol.atNums,[1,8,1,1]], dtype=np.int64)

    res = ab.computeDescriptorBatch(coords, atTypes)
    npt.assert_array_almost_equal(desc[1][0].reshape(-1).numpy(), res[1][0], 3)  # H
    npt.assert_array_almost_equal(desc[8][0].reshape(-1).numpy(), res[1][1], 3)  # C


def test_Bump3AngularDesc():
    """
    Model compounds atom idnexes are same as x/y coordinates:
            0,0,0, 1,0,0, 0,2,0, -3,0,0
    diagonal distances are 1-2: 2.3,    2-3: 3.6

                  2
                  |
                  |
    -3 ---------- 0 -----1
    """

    #                      atomTypes, nAngles, nRadial, rMin, rMax, aCent,rCent, halfWidth, maxWidthMultiplier, angleHalfWidth, maxAngleWidthMultiplier
    ab = Bump3AngularBasis([1,8],     3,       3,       1,    3,     None, None, 1,         5,                  0.4286,         8)
    mol   = PTMol(GDBMol(demoMols['angleTest']))

    # rShift      1       2       3        1       2      3        1       2      3
    # aShift      0       0       0       pi/2    pi/2    pi/2     pi      pi     pi     Angles
    expectH1 = [  0.0000, 0.0070, 0.0018, 0.0002, 0.0346, 0.0089, 0.0000, 0.0000, 0.0000,  # H(-H)-H
                  0.1229, 0.0424, 0.0111, 0.2165, 0.1935, 0.0387, 0.0002, 0.0002, 0.0000,  # H(-H)-O
                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000 ] # H(-O)-O
    expectO0 = [  0.0060, 0.0086, 0.0030, 0.3493, 0.4979, 0.1752, 0.2034, 0.0252, 0.2004,  # O(-H)-H
                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,  # O(-H)-O
                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000 ] # O(-O)-O


    # single conf
    coords = np.asarray([mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums], dtype=np.int64)
    res = ab.computeDescriptorBatch(coords, atTypes)

    npt.assert_array_almost_equal(expectH1, res[0][1], 3)  # H
    npt.assert_array_almost_equal(expectO0, res[0][0], 3)  # C


    # two confs
    coords = np.asarray([mol.baseMol.xyz,mol.baseMol.xyz], dtype=np.float32)
    atTypes = np.asarray([mol.atNums,mol.atNums], dtype=np.int64)
    res = ab.computeDescriptorBatch(coords, atTypes)

    npt.assert_array_almost_equal(expectH1, res[0][1], 3)  # H
    npt.assert_array_almost_equal(expectO0, res[0][0], 3)  # C
    npt.assert_array_almost_equal(expectH1, res[1][1], 3)  # H
    npt.assert_array_almost_equal(expectO0, res[1][0], 3)  # C

    # test order independence, swap C0 of second mol with H1
    cPos = coords[1,0].copy()
    coords[1,0] = coords[1,1]
    coords[1,1] = cPos
    atTypes = np.asarray([mol.atNums,[1,8,1,1]], dtype=np.int64)

    res = ab.computeDescriptorBatch(coords, atTypes)
    npt.assert_array_almost_equal(expectH1, res[1][0], 3)  # H
    npt.assert_array_almost_equal(expectO0, res[1][1], 3)  # C



def test_BumpAngularBasis1():

    #                     atomTypes, nAngles, nRadial, rMin, rMax, aCent,rCent, halfWidth, maxWidthMultiplier, angleHalfWidth, maxAngleWidthMultiplier
    ab = BumpAngularBasis([1,6],     3,       3,       1,    3,     None, None, 1,         3,                  0.4286,         3)
    gMol = PTMol(GDBMol(_molRadTest, quiet=True) )
    desc, idx1o, idx2o, idx3o = ab._computeAngularDescriptors(gMol)


    # rShift  1       2        3         1        2       3        1        2       3
    # aShift  0       0        0        pi/2     pi/2    pi/2      pi       pi      pi     Angles
    expect = np.array(
        [[ 0.5000,  0.5000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000], # 0-2, 0-1
         [ 0.0118,  1.0000,  0.0118,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000], # 0-3, 0-1
         [ 0.0000,  0.5000,  0.5000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000], # 0-3, 0-2
         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000, 0.0118,  0.0000], # 1-2, 1-0
         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.5000, 0.5000,  0.0000], # 1-3, 1-0
         [ 0.5000,  0.5000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000], # 1-3, 1-2
         [ 0.5000,  0.5000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000], # 2-1, 2-0
         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.5000, 0.5000,  0.0000], # 2-3, 2-0
         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000, 0.0118,  0.0000], # 2-3, 2-1
         [ 0.0000,  0.5000,  0.5000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000], # 3-1, 3-0
         [ 0.0118,  1.0000,  0.0118,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000], # 3-2, 3-0
         [ 0.5000,  0.5000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000]])# 3-2, 3-0
    expectCenter = [ 0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3]
    expectAt2    = [ 2,  3,  3,  2,  3,  3,  1,  3,  3,  1,  2,  2]
    expectAt3    = [ 1,  1,  2,  0,  0,  2,  0,  0,  1,  0,  0,  1]

    npt.assert_array_almost_equal(desc.numpy(), expect, 3)
    npt.assert_allclose(idx1o.numpy(), expectCenter)
    npt.assert_allclose(idx2o.numpy(), expectAt2)
    npt.assert_allclose(idx3o.numpy(), expectAt3)



def test_BumpAngularBasis2():
    """
    Model compounds atom idnexes are same as x/y coordinates:
            0,0,0, 1,0,0, 0,2,0, -3,0,0
    diagonal distances are 1-2: 2.3,    2-3: 3.6

                  2
                  |
                  |
    -3 ---------- 0 -----1
    """

    #                     atomTypes, nAngles, nRadial, rMin, rMax, aCent,rCent, halfWidth, maxWidthMultiplier, angleHalfWidth, maxAngleWidthMultiplier
    ab = BumpAngularBasis([1,6],     3,       3,       1,    3,     None, None, 1,         5,                  0.4286,         8)
    mol = PTMol(GDBMol(demoMols['angleTest']))
    desc, idx1o, idx2o, idx3o = ab._computeAngularDescriptors(mol)

    # rShift  1       2        3         1        2       3        1        2       3
    # aShift  0       0        0        pi/2     pi/2    pi/2      pi       pi      pi     Angles
    expect = np.array(
       [[ 0.0086,  0.0086,  0.0000,  0.5000,  0.5000,  0.0001,  0.0086, 0.0086,  0.0000],  # 0-2, 0-1
        [ 0.0000,  0.0000,  0.0000,  0.0007,  0.0173,  0.0007,  0.0421, 1.0000,  0.0421],  # 0-3, 0-1
        [ 0.0000,  0.0086,  0.0086,  0.0001,  0.5000,  0.5000,  0.0000, 0.0086,  0.0086],  # 0-3, 0-2
        [ 0.0493,  0.0979,  0.0001,  0.2444,  0.4850,  0.0005,  0.0002, 0.0005,  0.0000],  # 1-2, 1-0
        [ 0.0001,  0.5000,  0.5000,  0.0000,  0.0086,  0.0086,  0.0000, 0.0000,  0.0000],  # 1-3, 1-0
        [ 0.0000,  0.0023,  0.1403,  0.0000,  0.0113,  0.6955,  0.0000, 0.0000,  0.0007],  # 1-3, 1-2
        [ 0.0113,  0.6955,  0.0678,  0.0023,  0.1403,  0.0137,  0.0000, 0.0000,  0.0000],  # 2-1, 2-0
        [ 0.0000,  0.0328,  0.2002,  0.0000,  0.0872,  0.5321,  0.0000, 0.0000,  0.0002],  # 2-3, 2-0
        [ 0.0000,  0.0025,  0.0330,  0.0000,  0.0718,  0.9608,  0.0000, 0.0006,  0.0081],  # 2-3, 2-1
        [ 0.0000,  0.0001,  0.5000,  0.0000,  0.0000,  0.0086,  0.0000, 0.0000,  0.0000],  # 3-1, 3-0
        [ 0.0000,  0.0012,  0.4610,  0.0000,  0.0005,  0.1735,  0.0000, 0.0000,  0.0000],  # 3-2, 3-0
        [ 0.0000,  0.0000,  0.0872,  0.0000,  0.0000,  0.0328,  0.0000, 0.0000,  0.0000]]) # 3-2, 3-1

    expectCenter = [ 0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3]
    expectAt2    = [ 2,  3,  3,  2,  3,  3,  1,  3,  3,  1,  2,  2]
    expectAt3    = [ 1,  1,  2,  0,  0,  2,  0,  0,  1,  0,  0,  1]

    npt.assert_array_almost_equal(desc.numpy(), expect, 3)
    npt.assert_allclose(idx1o.numpy(), expectCenter)
    npt.assert_allclose(idx2o.numpy(), expectAt2)
    npt.assert_allclose(idx3o.numpy(), expectAt3)


def test_GaussianAngularBasis1():

    #                         atomTypes, nAngles, nRadial, rMin, rMax, aCent,rCent, halfWidth, angleHalfWidth, cutoff
    ab = GaussianAngularBasis([1,6],     3,       3,       1,    3,     None, None, 1,         0.4286,         9)
    gMol = PTMol(GDBMol(_molRadTest, quiet=True) )
    desc, idx1o, idx2o, idx3o = ab._computeAngularDescriptors(gMol)



    # rShift  1       2        3         1        2       3        1        2       3
    # aShift  0       0        0        pi/2     pi/2    pi/2      pi       pi      pi     Angles
    expect = np.array(
        [[ 0.8564,  0.8564,  0.0033,  0.0134,  0.0134,  0.0001,  0.0000, 0.0000,  0.0000], # 0-2, 0-1
         [ 0.0909,  1.4548,  0.0909,  0.0014,  0.0227,  0.0014,  0.0000, 0.0000,  0.0000], # 0-3, 0-1
         [ 0.0026,  0.6623,  0.6623,  0.0000,  0.0103,  0.0103,  0.0000, 0.0000,  0.0000], # 0-3, 0-2
         [ 0.0000,  0.0000,  0.0000,  0.0294,  0.0018,  0.0000,  1.8812, 0.1176,  0.0000], # 1-2, 1-0
         [ 0.0000,  0.0000,  0.0000,  0.0134,  0.0134,  0.0001,  0.8564, 0.8564,  0.0033], # 1-3, 1-0
         [ 0.8564,  0.8564,  0.0033,  0.0134,  0.0134,  0.0001,  0.0000, 0.0000,  0.0000], # 1-3, 1-2
         [ 0.8564,  0.8564,  0.0033,  0.0134,  0.0134,  0.0001,  0.0000, 0.0000,  0.0000], # 2-1, 2-0
         [ 0.0000,  0.0000,  0.0000,  0.0134,  0.0134,  0.0001,  0.8564, 0.8564,  0.0033], # 2-3, 2-0
         [ 0.0000,  0.0000,  0.0000,  0.0294,  0.0018,  0.0000,  1.8812, 0.1176,  0.0000], # 2-3, 2-1
         [ 0.0026,  0.6623,  0.6623,  0.0000,  0.0103,  0.0103,  0.0000, 0.0000,  0.0000], # 3-1, 3-0
         [ 0.0909,  1.4548,  0.0909,  0.0014,  0.0227,  0.0014,  0.0000, 0.0000,  0.0000], # 3-2, 3-0
         [ 0.8564,  0.8564,  0.0033,  0.0134,  0.0134,  0.0001,  0.0000, 0.0000,  0.0000]])# 3-2, 3-0
    expectCenter = [ 0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3]
    expectAt2    = [ 2,  3,  3,  2,  3,  3,  1,  3,  3,  1,  2,  2]
    expectAt3    = [ 1,  1,  2,  0,  0,  2,  0,  0,  1,  0,  0,  1]

    npt.assert_array_almost_equal(desc.numpy(), expect, 3)
    npt.assert_allclose(idx1o.numpy(), expectCenter)
    npt.assert_allclose(idx2o.numpy(), expectAt2)
    npt.assert_allclose(idx3o.numpy(), expectAt3)



def test_GaussianAngularBasis2():
    """
    Model compounds atom idnexes are same as x/y coordinates:
            0,0,0, 1,0,0, 0,2,0, -3,0,0
    diagonal istances are 1-2: 2.3,    2-3: 3.6

                  2
                  |
                  |
    -3 ---------- 0 -----1
    """

    #                         atomTypes, nAngles, nRadial, rMin, rMax, aCent,rCent, halfWidth, angleHalfWidth, cutoff
    ab = GaussianAngularBasis([1,6],     3,       3,       1,    3,     None, None, 1,        0.4286,         9)
    mol = PTMol(GDBMol(demoMols['angleTest']))
    desc, idx1o, idx2o, idx3o = ab._computeAngularDescriptors(mol)

    # rShift  1       2        3         1        2       3        1        2       3
    # aShift  0       0        0        pi/2     pi/2    pi/2      pi       pi      pi     Angles
    expect = np.array(
       [[ 0.0134,  0.0134,  0.0001,  0.8564,  0.8564,  0.0033,  0.0134, 0.0134,  0.0001],  # 0-2, 0-1
        [ 0.0000,  0.0000,  0.0000,  0.0014,  0.0227,  0.0014,  0.0909, 1.4548,  0.0909],  # 0-3, 0-1
        [ 0.0000,  0.0103,  0.0103,  0.0026,  0.6623,  0.6623,  0.0000, 0.0103,  0.0103],  # 0-3, 0-2
        [ 0.0826,  0.1589,  0.0012,  0.4155,  0.7995,  0.0060,  0.0003, 0.0005,  0.0000],  # 1-2, 1-0
        [ 0.0022,  0.5691,  0.5691,  0.0000,  0.0089,  0.0089,  0.0000, 0.0000,  0.0000],  # 1-3, 1-0
        [ 0.0000,  0.0045,  0.1386,  0.0000,  0.0227,  0.6975,  0.0000, 0.0000,  0.0004],  # 1-3, 1-2
        [ 0.0341,  1.0496,  0.1262,  0.0068,  0.2086,  0.0251,  0.0000, 0.0000,  0.0000],  # 2-1, 2-0
        [ 0.0000,  0.0427,  0.2286,  0.0001,  0.1142,  0.6122,  0.0000, 0.0000,  0.0001],  # 2-3, 2-0
        [ 0.0000,  0.0034,  0.0346,  0.0000,  0.1041,  1.0735,  0.0000, 0.0008,  0.0078],  # 2-3, 2-1
        [ 0.0000,  0.0017,  0.4401,  0.0000,  0.0000,  0.0069,  0.0000, 0.0000,  0.0000],  # 3-1, 3-0
        [ 0.0000,  0.0052,  0.4492,  0.0000,  0.0020,  0.1678,  0.0000, 0.0000,  0.0000],  # 3-2, 3-0
        [ 0.0000,  0.0001,  0.0759,  0.0000,  0.0000,  0.0283,  0.0000, 0.0000,  0.0000]]) # 3-2, 3-1

    expectCenter = [ 0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3]
    expectAt2    = [ 2,  3,  3,  2,  3,  3,  1,  3,  3,  1,  2,  2]
    expectAt3    = [ 1,  1,  2,  0,  0,  2,  0,  0,  1,  0,  0,  1]

    npt.assert_array_almost_equal(desc.numpy(), expect, 3)
    npt.assert_allclose(idx1o.numpy(), expectCenter)
    npt.assert_allclose(idx2o.numpy(), expectAt2)
    npt.assert_allclose(idx3o.numpy(), expectAt3)


def test_GaussianAngularBasis_descriptors():
    #                         atomTypes, nAngles, nRadial, rMin, rMax, aCent,rCent, halfWidth, angleHalfWidth, cutoff
    ab = GaussianAngularBasis([1,8],     3,       3,       1,    3,    None, None,  1,         0.4286,         9)
    mol = PTMol(GDBMol(demoMols['angleTest']))
    res = ab.computeDescriptors(mol)


    # [ H1 [[ H(-H)-H ] , [H(-O)-H],[H(-O)-O] ]],
    #   H2 [[ H(-H)-H ] , [H(-O)-H],[H(-O)-O] ]],
    #   H3 [[ H(-H)-H ] , [H(-O)-H],[H(-O)-O] ]]]
    expectH = \
        [[[ 0.0000,  0.0045,  0.1386,  0.0000,  0.0227,  0.6975,  0.0000, 0.0000,  0.0004],
          [ 0.0848,  0.7280,  0.5703,  0.4156,  0.8084,  0.0149,  0.0003, 0.0005,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000]],

         [[ 0.0000,  0.0034,  0.0346,  0.0000,  0.1041,  1.0735,  0.0000, 0.0008,  0.0078],
          [ 0.0341,  1.0922,  0.3549,  0.0069,  0.3228,  0.6373,  0.0000, 0.0000,  0.0001],
          [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000]],

         [[ 0.0000,  0.0001,  0.0759,  0.0000,  0.0000,  0.0283,  0.0000, 0.0000,  0.0000],
          [ 0.0000,  0.0070,  0.8893,  0.0000,  0.0020,  0.1746,  0.0000, 0.0000,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000]]]

    # [ O0 [[ O(-H)-H ] , [O(-O)-H],[O(-O)-O] ]],
    expectO = \
        [[[ 0.0134,  0.0237,  0.0104,  0.8604,  1.5414,  0.6670,  0.1043, 1.4785,  0.1013],
          [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000]]]

    npt.assert_array_almost_equal(res[8].numpy(), expectO, 3)
    npt.assert_array_almost_equal(res[1].numpy(), expectH, 3)


def test_angleBatch():
    coords = torch.tensor([[[0.,0,0],[1,0,0],[0,1,0]]])
    #                     atomTypes, nAngles, nRadial, rMin, rMax, aCent,rCent, halfWidth, angleHalfWidth, cutoff
    ab = GaussianAngularBasis([1,8], 3,       3,       1,    3,    None, None,  1,         0.4286,         9)
    angInfo = ab._computeAnglesBatch(1,3,torch.tensor([[8,1,1]]),coords)

    expectedA = [ 1.5708,  0.7854,  0.7854]
    expectedI = [ 0,1,2 ]
    expectedJ = [ 1,1,1 ]
    expectedK = [ 1,8,8 ]
    expectedDIJ = [1, 1.4142, 1.4142]
    expectedDIK = [1.,1,1]

    npt.assert_array_almost_equal(torch.acos(angInfo.cosAngles),  expectedA, 3)
    npt.assert_array_almost_equal(angInfo.distIJ, expectedDIJ, 3)
    npt.assert_array_almost_equal(angInfo.distIK, expectedDIK, 3)
    npt.assert_equal(angInfo.i_triple2Coords.numpy(), expectedI)
    npt.assert_equal(angInfo.j_AtType.numpy(), expectedJ)
    npt.assert_equal(angInfo.k_AtType.numpy(), expectedK)


    coords = torch.tensor([[[0.,0,0],[1,0,0],[0,1,0]],
                           [[1,0,0],[0.,0,0],[0,1,0]]])
    atTypes = torch.tensor([[8,1,1],[8,1,1]])
    angInfo = ab._computeAnglesBatch(2,3,atTypes, coords)

    expectedA = [ 1.5708,  0.7854,  0.7854, 0.7854,  1.5708,  0.7854]
    expectedI = [0, 1, 2, 3, 4, 5]
    expectedJ = [1, 1, 1, 1, 1, 1]
    expectedK = [1, 8, 8, 1, 8, 8]
    expectedDIJ = [1.0000, 1.4142, 1.4142,    1.4142, 1.0000, 1.000 ]
    expectedDIK = [1.0000, 1.0000, 1.0000,    1.0000, 1.0000, 1.4142 ]

    npt.assert_array_almost_equal(torch.acos(angInfo.cosAngles),  expectedA, 3)
    npt.assert_array_almost_equal(angInfo.distIJ.numpy(), expectedDIJ, 3)
    npt.assert_array_almost_equal(angInfo.distIK.numpy(), expectedDIK, 3)
    npt.assert_equal(angInfo.i_triple2Coords.numpy(), expectedI)
    npt.assert_equal(angInfo.j_AtType.numpy(), expectedJ)
    npt.assert_equal(angInfo.k_AtType.numpy(), expectedK)


    # cutoff
    ab = GaussianAngularBasis([1,8], 3,       3,       1,    3,    None, None,  1,         0.4286,         1.001)
    angInfo = ab._computeAnglesBatch(2,3,atTypes, coords)

    expectedA = [ 1.5708,  1.5708 ]
    expectedI = [0, 4]
    expectedJ = [1, 1]
    expectedK = [1, 8]
    expectedDIJ = [1., 1. ]
    expectedDIK = [1., 1. ]

    npt.assert_array_almost_equal(torch.acos(angInfo.cosAngles),  expectedA, 3)
    npt.assert_array_almost_equal(angInfo.distIJ.numpy(), expectedDIJ, 3)
    npt.assert_array_almost_equal(angInfo.distIK.numpy(), expectedDIK, 3)
    npt.assert_equal(angInfo.i_triple2Coords.numpy(), expectedI)
    npt.assert_equal(angInfo.j_AtType.numpy(), expectedJ)
    npt.assert_equal(angInfo.k_AtType.numpy(), expectedK)



# def test_angleBatch2():
#     coords = torch.tensor([[[1.,0,0],[0,2,0],[0,0,3]]])
#     #                         atomTypes, nAngles, nRadial, rMin, rMax, aCent,rCent, halfWidth, angleHalfWidth, cutoff
#     ab2 = GaussianAngularBasis([1,8],    3,       3,       1,    3,    None, None,  1,         0.4286,         3)
#     atTypes = torch.LongTensor([[1,1,1]])
#     res = ab2.computeDescriptorBatch(coords, atTypes)
#
#     expectedA = [ 1.5708,  0.7854,  0.7854]
#     expectedD = [[0.,1,1], [1,0,1.4142], [1,1.4142,0]]
#     expectedI = [ 0,1,2 ]
#     expectedJ = [ 2,2,1 ]
#     expectedK = [ 1,0,0 ]
#
