#!/bin/env python
"""
Created on May 14, 2018

@author: albertgo
"""

import gc
from ase.units import Hartree, kcal
from ase.units import mol as mol_unit
import ANI.lib.pyanitools as pya
from typing import Dict, Set, Sequence, Tuple, Optional, List, Any
import torch
import glob
import gzip
import re
import os
import os.path as path
import numpy as np
import logging
import h5py
from cdd_chem.util import constants
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611
from t_opt.atom_info import NameToNum
from ml_qm.ANIMol import ANIMol
from ml_qm import AtomInfo
from ml_qm.pt.nn.Precision_Util import NNP_PRECISION
log = logging.getLogger(__name__)




#torch.set_printoptions(linewidth=200)


class MEMDataSet():
    """
        Container for in memory storage of conformations.

        A MemDataSet holds all coordinates for all atoms of all conformations in CPU memory.

    """


    def __init__(self, debug=False):
        """
        MEMDataSet loads all coordinates into memory. and stores them in the internal fields


        Attributes:
        -----------
        allCoords['train'][atIdx]:           xyz coordinate of atIdx
        confIdx2MolIdx['train'][confIdx]:    molIdx of confIdx
        confIdx2FirstAtom['train'][confIdx]: is index to first atom of confIdx in allCoords
        molIdx2AtomTypes[molIdx]:            array of atomic numbers
        molIdx2Smiles[molIdx]:               smiles of molIdx
        energies['train'][confIdx]           energy of confIdx
        """

        # indexed by type train, test, val
        self.allCoords        = {}   # xyz coordinate of atIdx
                                     # atIdx is index over all atoms in all confs of all mols in all sets
        self.confIdx2FirstAtom= {}   # confIdx2FirstAtom['train'][confIdx] is index to first atom of confIdx
        self.confIdx2MolIdx   = {}   # confIdx2MolIdx['train'][confIdx] is molIdx of this conf
        self.molIdx2AtomTypes = []   # molIdx2AtomTypes[molIdx] is list of atomic numbers of atoms in molIdx
        self.molIdx2Smiles    = []   # molIdx2Smiles[molIdx] is smiles of molIdx
        self.energies         = {}   # energies['train'][confIdx] is tuple of energy in kcal_mol of confIdx, this may be Null for eval mode
        self.debug            = debug


    def _printMolInfo(self, smiles, allCoordsLists, confIdx2FirstAtomLists, confIdx2MolIdxLists, energiesLists):

        log.warning("\n------------------------------ %s", smiles)
        log.warning("molIdx2AtomTypes       %s", self.molIdx2AtomTypes)
        log.warning("allCoordsLists         %s", allCoordsLists)
        log.warning("confIdx2FirstAtomLists %s", confIdx2FirstAtomLists)
        log.warning("confIdx2MolIdxLists    %s", confIdx2MolIdxLists)
        log.warning("energiesLists          %s", energiesLists)


    def getSetLen(self, setType="train"):
        return self.confIdx2FirstAtom[setType].shape[0]


#     def __enter__(self):
#         return self
#
#     def __exit__(self, type, value, traceback):
#         pass;

    def _load_Files_ANI_1(self, setTypes:Sequence[str], inPattern:str, skipFactor:int,
                          maxEnergy_kcal_mol:float, maxEnergylowMW_kcalM:float,
                          energyCorrections:Dict[int,float],
                          valSmiles:Optional[Set[str]], testSmiles:Optional[Set[str]]):
        # pylint: disable=R0915
        atomIdx:                Dict[str,int]              = {}
        allCoordsLists:         Dict[str,List[np.ndarray]] = {}
        confIdx2FirstAtomLists: Dict[str,List[np.ndarray]] = {}
        confIdx2MolIdxLists:    Dict[str,List[int]]        = {}
        energiesLists:          Dict[str,List[float]]      = {}

        for setType in setTypes:
            allCoordsLists[setType]              = []
            confIdx2FirstAtomLists[setType]      = []
            confIdx2MolIdxLists[setType]         = []
            energiesLists[setType]               = []

            atomIdx[setType] = 0
            molIdx = 0

        sumHighE = 0
        for f in glob.glob(inPattern):
            log.info("Processing: %s", f)

            adl = pya.anidataloader(f)
            for rec in adl:

                # one record is n confomration of same molecule
                # The molecule has these atom types
                atomTypes = list(AtomInfo.NameToNum[a] for a in rec['species'])
                nAtomsPerMol = len(atomTypes)
                nHeavy = sum( 1 for at in atomTypes if at != 1 )

                # Extract the data
                smiles    = "".join(rec['smiles'])
                if False: print(smiles)
                coordsMol = rec['coordinates']   # nConf nAtoms x xyz
                e         = rec['energies']      # vector of energies x nConf
                e = e * Hartree*mol_unit/kcal

                maxE = maxEnergy_kcal_mol if nHeavy > 3 else maxEnergylowMW_kcalM
                minE   = e.min()
                isLowE = (e-minE) <= maxE
                sumHighE += len(e) - isLowE.sum()
                coordsMol = coordsMol[isLowE]
                e = e[isLowE]

                coordsMol = coordsMol[::skipFactor]
                e         = e[::skipFactor]

                # compute atomization energy in [kcal/mol]
                e -= sum( [ ANIMol.atomEnergies[atNum] for atNum in atomTypes ])
                if energyCorrections is not None:
                    e -= sum( [ energyCorrections[atNum] for atNum in atomTypes ])

                #if len(e) == 0: continue

                # set set type to train or val
                setType = self._getType(nHeavy, smiles, valSmiles, testSmiles)

                nConfsMol = coordsMol.shape[0]
                coordsMol = coordsMol.reshape(-1,3)

                self.molIdx2AtomTypes.append(atomTypes)
                self.molIdx2Smiles.append(smiles)
                allCoordsLists[setType].append(coordsMol)
                confIdx2FirstAtomLists[setType].append(np.arange(atomIdx[setType],atomIdx[setType] + nAtomsPerMol*nConfsMol, nAtomsPerMol, dtype=np.int64))
                confIdx2MolIdxLists[setType].append([molIdx] * nConfsMol) #type: ignore

                energiesLists[setType].append(e)

                molIdx += 1
                atomIdx[setType] += nAtomsPerMol * nConfsMol

                if self.debug:
                    self._printMolInfo(smiles, allCoordsLists, confIdx2FirstAtomLists, confIdx2MolIdxLists, energiesLists)

                if self.debug:  # debug
                    name = rec['path']
                    m = re.search('/([^/]+)', name)
                    assert m is not None
                    name = m.group(1)

        for setType in setTypes:
            if len(energiesLists[setType]) > 0:
                self.allCoords[setType]         = np.concatenate(allCoordsLists[setType]) \
                                                    .astype(dtype=np.float32, casting='same_kind') # one row per atom
                self.confIdx2FirstAtom[setType] = np.concatenate(confIdx2FirstAtomLists[setType])\
                                                    .astype(dtype=np.int64,   casting='same_kind') # one row per conf
                self.confIdx2MolIdx[setType]    = np.concatenate(confIdx2MolIdxLists[setType])\
                                                    .astype(dtype=np.int64,   casting='same_kind')
                self.energies[setType]          = np.concatenate(energiesLists[setType])\
                                                    .astype(dtype=NNP_PRECISION.npLossDType, casting='same_kind')
            else:
                self.allCoords[setType]         = np.empty(0,dtype=np.float32)
                self.confIdx2FirstAtom[setType] = np.empty(0,dtype=np.int64)
                self.confIdx2MolIdx[setType]    = np.empty(0,dtype=np.int64)
                self.energies[setType]          = np.empty(0,dtype=NNP_PRECISION.npLossDType)

            log.info("Completed reading files of %5s nConf=%i\tdropedHighE=%i nCoords = %s (skipfactor=%i)",
                 setType, self.energies[setType].shape[0], sumHighE, self.allCoords[setType].shape[0], skipFactor )

            gc.collect()
            gc.collect()


    def _getType(self, nAtomsPerMol:int, smiles:str, valSmiles:Optional[Set[str]], testSmiles:Optional[Set[str]] ) -> str:
        """ establish type of molecule from data """

        smiles_b = smiles.encode('utf-8')

        assert valSmiles is not None # method may be overwritten to deal with None
        assert testSmiles is not None # method may be overwritten to deal with None

        typ = "train"
        if nAtomsPerMol > 3:
            if smiles_b in valSmiles:
                typ = "val"
            elif smiles_b in testSmiles:
                typ = 'test'
        return typ




class MEMSingleDataSet(MEMDataSet):
    """ In Memory MEMDataSet for of single setType of set e.g 'train'. """

    def __init__(self, setType, energyKnown=False, debug=False):
        """
            In Memory MEMDataSetfor of single setType of set e.g 'train'.
            To fill this dataset use the addConformer method.

            Parameter:
            ----------
            setType: string, type of this set eg. ;'train'
            energyKnown: if True energies are known and stored for this data set


            molIdx and confIdx are the same as each call to addConformer increases
            the conf and molIdx.
        """

        super().__init__(debug=debug)

        self.setType = setType

        self._allCoordsArray:List[np.ndarray]  = []   # list of arrays containing coordinates, deleted after collectConformers

        # confIdx2FirstAtom['train'][confIdx] is index to first atom of confIdx
        self.confIdx2FirstAtom[setType]: Dict[str,List[np.ndarray]] = []

        # confIdx2MolIdx['train'][confIdx] is molIdx of this conf
        self.confIdx2MolIdx[setType]: Dict[str,List[np.ndarray]] = []

        # energies['train'][confIdx] is tuple of energy in kcal_mol of confIdx, this may be Null for eval mode
        if energyKnown:
            self.energies[setType]: Optional[Dict[str,List[np.ndarray]]] = []
        else:
            self.energies = None

        # these will only be available after calling collectConformers();
        self.nAtoms = 0
        self.allCoords[setType] = None   # allCords['train'][atIdx] = xyz coordinate of atIdx
                                         # atIdx is index over all atoms in all confs of all mols


    def addConformer(self, xyz, atomicNums, smiles, energy=None):
        """ Add a conformewr to this dataset

            xyz: array of xyz coordinates for each atom
            atomicNums: np.array.int64 of atomic numbers for each atom

        """
        self.confIdx2MolIdx[self.setType].append(len(self.molIdx2AtomTypes))
        self.confIdx2FirstAtom[self.setType].append(self.nAtoms)
        self.nAtoms = self.nAtoms + len(xyz)
        self._allCoordsArray.append(xyz)
        self.molIdx2AtomTypes.append(atomicNums)
        self.molIdx2Smiles.append(smiles)
        if self.energies is not None: self.energies[self.setType].append(energy)


    def collectConformers(self):
        """ this needs to be called after one or more calls to addconformer and before using this as an iterator
        """

        self.confIdx2FirstAtom[self.setType] = np.array(self.confIdx2FirstAtom[self.setType], dtype=np.int64)
        self.confIdx2MolIdx[self.setType]    = np.array(self.confIdx2MolIdx[self.setType], dtype=np.int64)
        if self.energies is not None:
            self.energies[self.setType]      = np.array(self.energies[self.setType], dtype=NNP_PRECISION.lossDType)

        self.allCoords[self.setType]         = np.concatenate(self._allCoordsArray) \
                                                 .astype(dtype=np.float32, casting='same_kind')
        del self._allCoordsArray



class ANIMEMSplitDataSet(MEMDataSet):
    """
        DataSet that splits the ANI data into training, val and test set based on
        lists of smiles.

    """

    def __init__(self, debug=False):
        super().__init__(debug=debug)

        self.split_sets: Any

    @staticmethod
    def getEnergyCorrections(conf, atomNames):
        energyCorrections = conf.get('energyCorrections', None)
        atomNums = [AtomInfo.NameToNum[at] for at in atomNames]
        atomNum2ECorrection = [0.] * (max(atomNums) + 1)

        if energyCorrections is not None:
            for i, at in enumerate(atomNums):
                atomNum2ECorrection[at] = energyCorrections[i]

        return atomNum2ECorrection


    def load_ANI_files(self, atom_types:Sequence[str], conf:Dict):
        atomNum2ECorrection = ANIMEMSplitDataSet.getEnergyCorrections(conf, atom_types)

        dataDir:str
        if isinstance(conf['dataDir'], str):
            dataDir = conf['dataDir']
        else:
            for d in conf['dataDir']:
                d = constants.replace_consts_and_env(d)
                if os.path.isdir(d):
                    dataDir = d
                    break

        iFile = "%s/%s" % (dataDir, conf['iFile'])
        setTypes = ("train", "val", "test")

        setformat = conf.get('type', 'ANI-1')
        if setformat.startswith('ANI-2'):
            seed         = conf.get('seed', 42424242)
            valFraction  = conf['valFraction']
            testFraction = conf['testFraction']
            atomizationE_range = conf['atomizationERange']

            splitStrategy = conf.get('splitStrategy', None)
            if splitStrategy   == 'allOrNone':
                self.split_sets = self._split_sets_all_to_one
            elif splitStrategy == 'splitByFraction':
                self.split_sets = self._split_sets_split_sets
            else:
                raise NameError(f"Invalid trainData.conf for splitStrategy: {splitStrategy}")

            if setformat   == 'ANI-2_201910':
                self._load_Files_ANI_201910(setTypes, iFile, conf['skipFactor'],
                                       atomizationE_range, atomNum2ECorrection,
                                       seed = seed, valFraction=valFraction, testFraction=testFraction)
            elif setformat == 'ANI-2_202001':
                self._load_Files_ANI_202001(setTypes, iFile, conf['skipFactor'],
                                       atomizationE_range, atomNum2ECorrection,
                                       seed = seed, valFraction=valFraction, testFraction=testFraction)
            else:
                self._load_Files_ANI_2(setTypes, iFile, conf['skipFactor'],
                                       atomizationE_range, atomNum2ECorrection,
                                       seed = seed, valFraction=valFraction, testFraction=testFraction)

        else:
            maxEnergy = conf['maxEnergy_kcal']
            maxEnergyLowMW = conf.get('maxEnergyLowMW_kcal', 999 if maxEnergy < 40 else maxEnergy)
            valSmiFile = constants.replace_consts_and_env(conf["valSmilesFile"])
            tstSmiFile = constants.replace_consts_and_env(conf.get("tstSmilesFile", None))
            assert valSmiFile is not None
            assert tstSmiFile is not None
            valSmiles = set(str(line.strip()) for line in gzip.open(valSmiFile))
            tstSmiles = set(str(line.strip()) for line in gzip.open(tstSmiFile)) if tstSmiFile is not None else set()

            assert valSmiles is not None
            assert tstSmiles is not None

            self._load_Files_ANI_1(setTypes, iFile, conf['skipFactor'],
                                   maxEnergy, maxEnergyLowMW, atomNum2ECorrection,
                                   valSmiles, tstSmiles)


    def _split_sets_split_sets(self, rnd:np.random.RandomState, n_heavy:float,
                  coordsMol:np.ndarray, e:np.ndarray,
                  valFraction:float, testFraction:float) \
            -> Tuple[Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]], Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]]:
        """ This methods splits the coordsMol according to the val/test fraction values.

            return (train_coords, val_coords, test_coords), (train_e, val_e, test_e)

        """

        if n_heavy <= 3:
            return (coordsMol, None, None), (e, None, None)

        valFraction += testFraction
        rndVal  = rnd.rand(coordsMol.shape[0])
        is_train = rndVal >= valFraction
        is_test  = rndVal < testFraction
        is_val   = ~is_train & ~is_test

        return (coordsMol[is_train],coordsMol[is_val],coordsMol[is_test]) ,(e[is_train],e[is_val],e[is_test])


    def _split_sets_all_to_one(self, rnd:np.random.RandomState, n_heavy:float,
                  coordsMol:np.ndarray, e:np.ndarray,
                  valFraction:float, testFraction:float)  \
            -> Tuple[Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]], Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]]:
        """ This method returns only one set containing all coordsMol depending, the other two sets are None.
            return (train_coords, val_coords, test_coords), (train_e, val_e, test_e)
        """

        if n_heavy <= 3:
            return (coordsMol, None, None), (e, None, None)

        valFraction += testFraction
        rndVal = rnd.rand()

        if rndVal < testFraction:
            return (None, None, coordsMol), (None, None, e)
        if rndVal < valFraction:
            return (None, coordsMol, None), (None, e, None)

        return (coordsMol, None, None), (e, None, None)


    def _load_Files_ANI_2(self,
                          setTypes:Sequence[str], inPattern:str, skipFactor:int,
                          atomizationE_range: Sequence[float], energyCorrections:Dict[int,float],
                          seed:int = 42424242, valFraction:float = 0.1, testFraction:float = 0.):
        # pylint: disable=R0915
        """ Note that energy cutoffs do not make sense for ANI-2 because a group may
            contain isomers and multiple group may contain molecules with the same connectivity.
        """

        rnd = np.random.RandomState(seed)

        atomIdx:                Dict[str,int]              = {}
        allCoordsLists:         Dict[str,List[np.ndarray]] = {}
        confIdx2FirstAtomLists: Dict[str,List[np.ndarray]] = {}
        confIdx2MolIdxLists:    Dict[str,List[int]]        = {}
        energiesLists:          Dict[str,List[float]]      = {}


        for set_type in setTypes:
            allCoordsLists[set_type]         = []
            confIdx2FirstAtomLists[set_type] = []
            confIdx2MolIdxLists[set_type]    = []
            energiesLists[set_type]          = []

            atomIdx[set_type] = 0
            molIdx = 0

        sumHighE = 0
        files = glob.glob(inPattern)
        files.sort()
        for f in files:
            log.info("Processing: %s", f)

            adl = pya.anidataloader(f)
            f = path.split(f)[1]
            for rec in adl:

                atomTypes = list(AtomInfo.NameToNum[a] for a in rec['species'])
                nAtomsPerMol = len(atomTypes)
                nHeavy = sum( 1 for at in atomTypes if at != 1 )

                # Extract the data
                name      = rec['path']
                coordsMol = rec['coordinates']
                e         = rec['energies']
                e         = e * Hartree*mol_unit/kcal

                coordsMol = coordsMol[::skipFactor]
                e         = e[::skipFactor]

                # compute atomization energy in [kcal/mol]
                e -= sum( [ ANIMol.atomEnergies[atNum] for atNum in atomTypes ])
                if energyCorrections is not None:
                    e -= sum( [ energyCorrections[atNum] for atNum in atomTypes ])

                min_E, max_E = atomizationE_range
                is_good = e >= min_E
                for i, en in enumerate(e): is_good[i] = is_good[i] and en <= max_E
                e = e[is_good]
                if len(e) == 0: continue

                coordsMol = coordsMol[is_good]

                coord_by_set, e_by_set = self.split_sets(rnd, nHeavy, coordsMol, e, valFraction, testFraction)

                self.molIdx2AtomTypes.append(atomTypes)
                self.molIdx2Smiles.append(name)

                for set_type, coordsMol, e in zip(('train', 'val', 'test'), coord_by_set, e_by_set):
                    if e is None or e.shape[0] == 0: continue

                    nConfsMol = coordsMol.shape[0]
                    coordsMol = coordsMol.reshape(-1,3)

                    allCoordsLists[set_type].append(coordsMol)
                    confIdx2FirstAtomLists[set_type].append(
                        np.arange(atomIdx[set_type], atomIdx[set_type] + nAtomsPerMol*nConfsMol,
                                  nAtomsPerMol, dtype=np.int64))
                    confIdx2MolIdxLists[set_type].append([molIdx] * nConfsMol) # type: ignore

                    energiesLists[set_type].append(e)

                    atomIdx[set_type] += nAtomsPerMol * nConfsMol

                molIdx += 1

        for set_type in setTypes:
            if len(energiesLists[set_type]) > 0:
                self.allCoords[set_type]         = np.concatenate(allCoordsLists[set_type]) \
                                                    .astype(dtype=np.float32, casting='same_kind') # one row per atom
                self.confIdx2FirstAtom[set_type] = np.concatenate(confIdx2FirstAtomLists[set_type])\
                                                    .astype(dtype=np.int64,   casting='same_kind') # one row per conf
                self.confIdx2MolIdx[set_type]    = np.concatenate(confIdx2MolIdxLists[set_type])\
                                                    .astype(dtype=np.int64,   casting='same_kind')
                self.energies[set_type]          = np.concatenate(energiesLists[set_type])\
                                                    .astype(dtype=NNP_PRECISION.npLossDType, casting='same_kind')
            else:
                self.allCoords[set_type]         = np.empty(0,dtype=np.float32)
                self.confIdx2FirstAtom[set_type] = np.empty(0,dtype=np.int64)
                self.confIdx2MolIdx[set_type]    = np.empty(0,dtype=np.int64)
                self.energies[set_type]          = np.empty(0,dtype=NNP_PRECISION.npLossDType)

            log.info("Completed reading files of %5s nConf=%i\tdropedHighE=%i nCoords = %s (skipfactor=%i)",
                 set_type, self.energies[set_type].shape[0], sumHighE, self.allCoords[set_type].shape[0], skipFactor)

            gc.collect()
            gc.collect()


    def _load_Files_ANI_201910(self, setTypes:Sequence[str], inPattern:str, skipFactor:int,
                          atomizationE_range: Sequence[float], energyCorrections:Dict[str,float],
                          seed:int = 42424242, valFraction:float = 0.1, testFraction:float = 0.):

        self._load_Files_ANI_2_New( "atomic_numbers", None, "wb97x_dz.energy", "coordinates", "wb97x_dz.forces",
            setTypes, inPattern, skipFactor,
            atomizationE_range, energyCorrections,
            seed, valFraction, testFraction)


    def _load_Files_ANI_202001(self, setTypes:Sequence[str], inPattern:str, skipFactor:int,
                          atomizationE_range: Sequence[float], energyCorrections:Dict[str,float],
                          seed:int = 42424242, valFraction:float = 0.1, testFraction:float = 0.):

        self._load_Files_ANI_2_New( None, "species", "energies", "coordinates", "forces",
            setTypes, inPattern, skipFactor,
            atomizationE_range, energyCorrections,
            seed, valFraction, testFraction)


    def _load_Files_ANI_2_New(self, at_num_field:Optional[str], at_sym_field:Optional[str], energy_field:str, coord_field:str, force_field:str,
                          setTypes:Sequence[str], inPattern:str, skipFactor:int,
                          atomizationE_range: Sequence[float], energyCorrections:Dict[str,float],
                          seed:int = 42424242, valFraction:float = 0.1, testFraction:float = 0.):
        # pylint: disable=W0613,R0915
        """ Note that energy cutoffs do not make sense for ANI-2 because a group may
            contain isomers and multiple group may contain molecules with the same connectivity.
        """

        rnd = np.random.RandomState(seed)

        atomIdx:                Dict[str,int]             = {}
        allCoordsLists:         Dict[str,List[np.ndarray]]= {}
        confIdx2FirstAtomLists: Dict[str,List[np.ndarray]]= {}
        confIdx2MolIdxLists:    Dict[str,List[int]]       = {}
        energiesLists:          Dict[str,List[float]]     = {}

        for set_type in setTypes:
            allCoordsLists[set_type]         = []
            confIdx2FirstAtomLists[set_type] = []
            confIdx2MolIdxLists[set_type]    = []
            energiesLists[set_type]          = []

            atomIdx[set_type] = 0
            molIdx = 0

        sumHighE = 0
        files = glob.glob(inPattern)
        files.sort()
        for f in files:
            log.info("Processing: %s" , f)

            cnt_conf = 0
            inFile = h5py.File(f, "r")
            f = path.split(f)[1]
            for key,item in inFile.items():

                if at_num_field:
                    atomTypes = item[at_num_field][()] # np.array(uint8)
                else:
                    atomTypes = [ NameToNum[s.decode('ascii')] for s in item[at_sym_field][()] ] # np.array(uint8)

                nAtomsPerMol = len(atomTypes)
                nHeavy = sum( 1 for at in atomTypes if at != 1 )

                # Extract the data
                name      = str(key)
                coordsMol = item[coord_field][()] # np.array((nMol,nAt,3])
                e         = item[energy_field][()] # np.array((nMol,nAt])
                e         = e * Hartree*mol_unit/kcal

                nconf = e.shape[0]
                cnt_conf += nconf
                if cnt_conf // 400000 != (cnt_conf - nconf) // 400000:
                    warn(f"conf {cnt_conf}")

                coordsMol = coordsMol[::skipFactor]
                e         = e[::skipFactor]

                # compute atomization energy in [kcal/mol]
                e -= sum( [ ANIMol.atomEnergies[atNum] for atNum in atomTypes ])
                if energyCorrections is not None:
                    e -= sum( [ energyCorrections[atNum] for atNum in atomTypes ])

                min_E, max_E = atomizationE_range
                is_good=(e >= min_E) & (e <= max_E)
                e = e[is_good]
                if len(e) == 0: continue

                coordsMol = coordsMol[is_good]

                coord_by_set, e_by_set = self.split_sets(rnd, nHeavy, coordsMol, e, valFraction, testFraction)

                self.molIdx2AtomTypes.append(atomTypes)
                self.molIdx2Smiles.append(name)

                for set_type, coordsMol, e in zip(('train', 'val', 'test'), coord_by_set, e_by_set):
                    if e is None or e.shape[0] == 0: continue

                    nConfsMol = coordsMol.shape[0]
                    coordsMol = coordsMol.reshape(-1,3)

                    allCoordsLists[set_type].append(coordsMol)
                    confIdx2FirstAtomLists[set_type].append(
                        np.arange(atomIdx[set_type], atomIdx[set_type] + nAtomsPerMol*nConfsMol,
                                  nAtomsPerMol, dtype=np.int64))
                    confIdx2MolIdxLists[set_type].append([molIdx] * nConfsMol) #type: ignore

                    energiesLists[set_type].append(e)

                    atomIdx[set_type] += nAtomsPerMol * nConfsMol

                molIdx += 1

        for set_type in setTypes:
            if len(energiesLists[set_type]) > 0:
                self.allCoords[set_type]         = np.concatenate(allCoordsLists[set_type]) \
                                                    .astype(dtype=np.float32, casting='same_kind') # one row per atom
                self.confIdx2FirstAtom[set_type] = np.concatenate(confIdx2FirstAtomLists[set_type])\
                                                    .astype(dtype=np.int64,   casting='same_kind') # one row per conf
                self.confIdx2MolIdx[set_type]    = np.concatenate(confIdx2MolIdxLists[set_type])\
                                                    .astype(dtype=np.int64,   casting='same_kind')
                self.energies[set_type]          = np.concatenate(energiesLists[set_type])\
                                                    .astype(dtype=NNP_PRECISION.npLossDType, casting='same_kind')
            else:
                self.allCoords[set_type]         = np.empty(0,dtype=np.float32)
                self.confIdx2FirstAtom[set_type] = np.empty(0,dtype=np.int64)
                self.confIdx2MolIdx[set_type]    = np.empty(0,dtype=np.int64)
                self.energies[set_type]          = np.empty(0,dtype=NNP_PRECISION.npLossDType)

            log.info("Completed reading files of %5s nConf=%i\tdropedHighE=%i nCoords = %s (skipfactor=%i)",
                 set_type, self.energies[set_type].shape[0], sumHighE, self.allCoords[set_type].shape[0], skipFactor)

            gc.collect()
            gc.collect()


class ANIMEMSingleDataSet(MEMDataSet):
    """
        Same as ANIMEMSplitDataSet but loads data for a single setType e.g. 'train'.
        Read from hd5 files in the ANI format
    """


    def __init__(self, setType:str, debug:bool = False):
        """
            inPattern: file name pattern used to identfy files to be loaded.
            setType: ONE OF 'train', 'val', test'
            skipFactor: to speed up testing, only every skipFactor conformation is loaded
            maxEnergy: only conformations with e < maxenergy relative to the lowest energy conformation are loaded

            energyCorrections: is an correction[atomicNumber] to be added to each energy.
        """
        super().__init__(debug=debug)
        self.setType = setType


    def load_ANI_files(self, inPattern:str, atom_types:Sequence[str], conf:Dict):

        atomNum2ECorrection = ANIMEMSplitDataSet.getEnergyCorrections(conf, atom_types)
        dataDir = constants.replace_consts_and_env(conf['dataDir'])
        iFile = "%s/%s" % (dataDir, inPattern)
        maxEnergy = conf['maxEnergy_kcal']

        setTypes = (self.setType,)
        self._load_Files_ANI_1(setTypes, iFile, conf['skipFactor'],
                               maxEnergy, maxEnergy, atomNum2ECorrection,
                               None, None )


    def _getType(self, nAtomsPerMol:int, smiles:str, valSmiles:Optional[Set[str]], testSmiles:Optional[Set[str]] ) -> str:
        # pylint: disable=W0613
        """ Overwrite so that all inputs are of self.setType """

        return self.setType



class TypedDataSet():
    """ TypedDataSetis a wrapper around ANIMEMSplitDataSet that exposes just a single
        dataset type 'train', 'val' ot 'test'

        Batches of conformations can be accessed is via the getCoordsBatch() method
    """

    def __init__(self, memDataSet, setType):
        self.dataSet = memDataSet
        self.setType = setType

    def __len__(self):
        return self.dataSet.getSetLen(self.setType)


    def getCoordsBatch(self, confIdxs, pinMem=False):
        molIdx2AtomTypes   = self.dataSet.molIdx2AtomTypes
        confIdx2MolIdx     = self.dataSet.confIdx2MolIdx[self.setType]
        confIdx2FirstAtom  = self.dataSet.confIdx2FirstAtom[self.setType]
        allCoords          = self.dataSet.allCoords[self.setType]
        energies = None
        if self.dataSet.energies is not None:
            energies       = self.dataSet.energies[self.setType]


        return CoordsBatch(self.setType, confIdxs, molIdx2AtomTypes, confIdx2MolIdx, confIdx2FirstAtom, allCoords, energies, pinMem)



class CoordsBatch():
    ''' A corrdsbatch represents a batch of conformations with all its coordinates.
        The Batch is split into conformations which have the same number of atoms to
        allow efficient computation of symmetry functions on the GPU.

        The nextCoordsByAtomCount() methods provides access to these subsets.

        Atom positions within a molecule are untouched.
    '''

    def __init__(self, setType, confIdxs, molIdx2AtomTypes, confIdx2MolIdx, confIdx2FirstAtom, allCoords, energies, pinMem=False):
        # pylint: disable=R0915
        """
           Store a batch of coordinates already split by atom count
        """
        #warn(confIdxs)
        if not isinstance(confIdxs, np.ndarray):
            confIdxs = np.array(confIdxs, dtype=np.int64)

        # confNum is internal to one batch 0 - conformatios in batch
        # confIdx referes to the index in the confIdx ad passed in

        self.nConfs = len(confIdxs)
        self.batchCoordsList :List[torch.tensor]     = []   # coordinates split to have all same atom count
        self.batchAtTypesList:List[torch.tensor]     = []   # atomTypes for each coordinate
        self.batchEnergies   :Optional[torch.tensor] = None # one per conf
        self.batchAt2ConfNum :torch.tensor                  # one per atom
        self.at2AtType       :torch.tensor                  # one per atom: convert coordinate position to unique atom type

        self.uAtTypes        :np.ndarray                    # list of unique atoms in this minibatch
        self.atTypeCount     :np.ndarray                    # count of atoms for each unique atom type

        assert confIdxs.dtype==np.int64, "confIdx must be np.int64"


        confLens = [len(molIdx2AtomTypes[confIdx2MolIdx[idx]]) for idx in confIdxs]

        # sort conformations by number of atoms
        sortByLenIdxs = np.argsort(confLens) #,kind='mergesort') # mergesort is slower but stable -> better for debugging
        confLens      = np.asarray(confLens)[sortByLenIdxs]
        confIdxs      = confIdxs[sortByLenIdxs]
        batchEnergies = None
        if energies is not None: batchEnergies = energies[confIdxs]

        # split into portions with same atom count
        atCntsPerLen, splitIdxs = np.unique(confLens, return_counts=True)
        splitIdxs = splitIdxs.cumsum()[:-1] # unique gives us the counts, cumsum gives us the split positions

        confIdxsByLen = np.split(confIdxs,splitIdxs)  # is is [ [indecesOfconfsWithSameLength] , ...]

        #######################################################################
        # split minibatch by atomCount in confs
        #######################################################################
        batchAt2ConfNumList = []
        startConfNum = 0
        for i,  nAt in enumerate(atCntsPerLen):

            #if nAt != 4: continue  #debug

            nAtConfIdxs = confIdxsByLen[i]

            # add one dimension [nConfs] -> [nConfs][nAtoms]
            # then add 0,1,..nAt to each
            atIdxs = confIdx2FirstAtom[nAtConfIdxs].reshape(-1,1) + np.arange(0,nAt)

            coords  = torch.from_numpy(allCoords[atIdxs].reshape(-1,nAt,3))
            atTypes = np.asarray([ molIdx2AtomTypes[confIdx2MolIdx[idx]] for idx in nAtConfIdxs],
                                 dtype=np.int64)
            atTypes = torch.from_numpy(atTypes)

            if pinMem:
                coords = coords.contiguous().pin_memory()
                atTypes = atTypes.contiguous().pin_memory()

            atIdx2ConfNum  = np.repeat(np.arange(0,len(nAtConfIdxs), dtype=np.int64)+startConfNum, nAt)
            startConfNum  += len(nAtConfIdxs)

            self.batchCoordsList.append( coords )
            batchAt2ConfNumList.append(atIdx2ConfNum )
            self.batchAtTypesList.append( atTypes )
            #################### batchAtTypesList.append(atTypes.reshape(-1))

        batchAtTypes = np.concatenate([atTyp.reshape(-1) for atTyp in self.batchAtTypesList])

        # get unique atom types so we can split by atom types
        self.uAtTypes, at2AtType, atTypeCount \
               = np.unique(batchAtTypes, return_inverse=True, return_counts=True)
        at2AtType = self.uAtTypes[at2AtType]
        self.atTypeCount = atTypeCount.tolist()


        batchAt2ConfNum = np.concatenate(batchAt2ConfNumList)
        if setType == 'train':
            # shuffle so that confnums are not size ordered:
            oldNum2New = np.int64(np.random.permutation(self.nConfs))
            newNum2Old = oldNum2New.argsort()
            batchAt2ConfNum = newNum2Old[batchAt2ConfNum]
            if batchEnergies is not None:
                self.batchEnergies = torch.from_numpy(batchEnergies[oldNum2New]).to(NNP_PRECISION.lossDType)
        elif batchEnergies is not None:
            self.batchEnergies = torch.from_numpy(batchEnergies).to(NNP_PRECISION.lossDType)

        # Now we are in the pytorch world
        ##### batchDesc         = torch.cat(batchDescList)
        self.at2AtType          = torch.from_numpy(at2AtType)
        self.batchAt2ConfNum    = torch.from_numpy(batchAt2ConfNum)

        if pinMem:
            self.at2AtType          = self.at2AtType.contiguous().pin_memory()
            self.batchAt2ConfNum    = self.batchAt2ConfNum.contiguous().pin_memory()
            if self.batchEnergies is not None:
                self.batchEnergies  = self.batchEnergies.contiguous().pin_memory()



    def nextCoordsByAtomCount(self):
        """ iterate over the sets of conformation in this batch by number of atoms """
        for coords, atTypes in zip(self.batchCoordsList,self.batchAtTypesList):
            yield coords, atTypes
