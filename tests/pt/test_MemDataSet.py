import pytest

import torch
import numpy as np
import numpy.testing as npt
import json

from cdd_chem.rdkit.io import MolInputStream
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611
from cdd_chem.util.constants import import_constants
from ml_qm.pt.nn.ani_net import ANINetInitializer
from ml_qm.pt.MEMBatchDataSet import MEMBatchDataSet, MultiBatchComputerDataSet
from ml_qm.pt.MEMDataSet import MEMSingleDataSet, ANIMEMSplitDataSet, TypedDataSet
cddCs = import_constants()

torch.set_printoptions(linewidth=200)


class TestMEMSingleDataSet:
    """ Test MEMSingleDataSet
    """
    def setup(self):
        jsonFile = 'scripts/trainHCON_Bump_48.json'
        with open(jsonFile) as jFile:
            self.conf = json.load(jFile)

        nGPU=1
        self.device = torch.device("cuda" if nGPU > 0 and torch.cuda.is_available() else "cpu")


        self.singleDS = MEMSingleDataSet('val')

        for sdName in ("tests/data/H2O.sdf", "tests/data/CH4.sdf", "tests/data/N2_long.sdf", "tests/data/Acetylene.sdf", "tests/data/CCCC.sdf"):
            with MolInputStream(sdName) as molS:
                for mol in molS:
                    xyz = mol.coordinates
                    atomicNumbers = mol.atom_types
                    smiles = mol.canonical_smiles

                    self.singleDS.addConformer(xyz,atomicNumbers, smiles)

        self.singleDS.collectConformers()
        assert self.singleDS.getSetLen('val') == 7
        initAni = ANINetInitializer(self.conf,self.device)
        self.descComputer = initAni.create_descriptor_computer()


    def test_DS(self):
        memBatchDS = MEMBatchDataSet(self.singleDS, 2000, self.descComputer, 'val', device=self.device)
        ds = MultiBatchComputerDataSet(memBatchDS, 2000, False)
        t=ds.next()
        next(t)  # noqa: F841

class TestMemBatch:
    """ Test MEMSingleDataSet """
    def setup(self):
        self.memDS = MEMSingleDataSet('val') # pylint: disable=W0201
        with MolInputStream("tests/data/N2_profile.sdf") as molIt:
            for i,mol in enumerate(molIt): # pylint: disable=W0612
                xyz = mol.coordinates
                atomTypes = np.array(mol.atom_types, dtype=np.int64)
                smiles = mol.canonical_smiles
                self.memDS.addConformer(xyz, atomTypes, smiles)
                #if i==18: break

        self.memDS.collectConformers()


        nGPU=1
        device = torch.device("cuda" if nGPU > 0 and torch.cuda.is_available() else "cpu")
        devId = ""
        if str(device) == "cuda":
            devId = torch.cuda.current_device()
        warn("nGPU=%i Device=%s %s" % (nGPU,device,devId))

        jsonFile = 'scripts/trainHCON_Bump_48.json'

        with open(jsonFile) as jFile:
            conf = json.load(jFile)
        initAni = ANINetInitializer(conf,device)
        descComputer = initAni.create_descriptor_computer()

        # pylint: disable=W0201
        self.memBatchDS = MEMBatchDataSet(self.memDS, 2000, descComputer,
                                          'val', True, device=device)



    def test_NNIsSorted(self):
        # the coordinates in the MEMSingleDataSet should be in input order
        # coordinates of NN distance are ordered from 1 to 6 A
        nnDist = self.memDS.allCoords['val'][:,0]
        nnDist=nnDist[nnDist!=0]
        nnDistSorted = nnDist
        nnDistSorted.sort()

        npt.assert_equal(nnDist,nnDistSorted)


class TestMemBatch2():
    """ TestMemBatch2 """
    def setup(self):
        self.confIDs = [1, 2, 0, 4, 3] # pylint: disable=W0201

        nGPU=1
        device = torch.device("cuda" if nGPU > 0 and torch.cuda.is_available() else "cpu")
        devId = ""
        if str(device) == "cuda":
            devId = torch.cuda.current_device()
        warn("nGPU=%i Device=%s %s" % (nGPU,device,devId))

        jsonFile = 'scripts/trainHCON_Bump_48.json'

#         valSmiles = set(line.strip() for line in gzip.open('data/all_val.txt.gz'))
#         tstSmiles = set(line.strip() for line in gzip.open('data/all_tst.txt.gz'))

        with open(jsonFile) as jFile:
            conf = json.load(jFile)
        initAni = ANINetInitializer(conf,device)
        descComputer = initAni.create_descriptor_computer()
#         dataDir = constants.replace_consts_and_env(conf['dataDir'])
#         iFile = "%s/%s" %(dataDir, conf['iFile'] )
#         skip = conf['skipFactor']
#         maxE = conf['maxEnergy_kcal']
        trainDataConf = conf['trainData']
        atom_types    = conf['atomTypes']

        memDS =  ANIMEMSplitDataSet(debug=False)
        memDS.load_ANI_files(atom_types, trainDataConf)

#         memDS =  ANIMEMSplitDataSet(iFile, valSmiles, tstSmiles, skip, maxE, debug=False)
        self.ds      = TypedDataSet(memDS, "train") # pylint: disable=W0201
        # pylint: disable=W0201
        self.batchDS = MEMBatchDataSet(memDS, 15, descComputer, 'train', True, device=device)


    def test_dataSet(self):
        # compute descriptors conformer by conformer and all together
        # compare that the sum of descriptors of each atom type is equal to validate
        # that the assembly of the descriptors by atom type is correct
        uAtomTypesList = []
        coorSum = {}
        uAtTypeCount = {}
        energies = []

        for conf in self.confIDs:
            cBatch = self.ds.getCoordsBatch([conf])
            uAtomTypesList.append(cBatch.uAtTypes)
            energies.append(cBatch.batchEnergies)

            for coords, atTypes in cBatch.nextCoordsByAtomCount():
                for coor, atT in zip(coords[0], atTypes[0]):
                    atT = atT.item()
                    if atT not in coorSum:
                        coorSum[atT] = 0
                    coorSum[atT] += coor.sum()

            for atT, atCount in zip(cBatch.uAtTypes, cBatch.atTypeCount):
                if atT not in uAtTypeCount:
                    uAtTypeCount[atT] = 0
                uAtTypeCount[atT] += atCount

        #########################################################
        cBatch = self.ds.getCoordsBatch(self.confIDs)
        assert pytest.approx(sum(energies).sum().cpu().item()) == cBatch.batchEnergies.sum().cpu().item()

        for i, (coor, atTypes) in enumerate(cBatch.nextCoordsByAtomCount()):
            for atT in cBatch.uAtTypes:
                coorSum[atT] -= coor[atTypes == int(atT)].sum()

        for i, atT in enumerate(cBatch.uAtTypes):
            npt.assert_allclose(coorSum[atT],np.array(0.),atol=1e-5)
            assert uAtTypeCount[atT] ==  cBatch.atTypeCount[i]
