## Alberto
import torch
import numpy as np

#torch.set_printoptions(linewidth=200)


class ANIDComputer():
    """
           This class computes ANI-1 like descriptors.
    """

    def __init__(self, atomTypes, radialBasis, angularBasis=None):
        """
           This class computes ANI-1 like descriptors.

           atomTypes -- list of atomic numbers supported
           radialBasis -- e.g. GaussianRadialBasis
           cutoff -- should be same as used to construct radialBasis

           Smith, J. S., O. Isayev, and A. E. Roitberg.
           ANI-1: An Extensible Neural Network Potential with DFT Accuracy at Force Field Computational Cost.
           Chem. Sci. 8, no. 4 (2017): 3192-3203


        """

        self.atomTypes = atomTypes
        self.radialBasis = radialBasis
        self.angularBasis = angularBasis


    def state_dict(self):
        destination = {}
        name = type(self).__name__
        destination["%s:atomTypes" % name] = self.atomTypes
        destination["%s:radialState"% name] = self.radialBasis.state_dict()
        if self.angularBasis is not None:
            destination["%s:angularState"% name] = self.angularBasis.state_dict()

        return destination

    def load_state_dict(self,state):
        name = type(self).__name__
        self.atomTypes = state["%s:atomTypes" % name]
        self.radialBasis.load_state_dict(state["%s:radialState" % name])
        if self.angularBasis is not None:
            self.angularBasis.load_state_dict(state["%s:angularState" % name])

    def printOptParam(self):
        self.radialBasis.printOptParam()
        if self.angularBasis is not None:
            self.angularBasis.printOptParam()


    def to(self, device):
        """ move internal tnesors to device """
        self.radialBasis.to(device)
        if self.angularBasis is not None:
            self.angularBasis.to(device)


    def eval(self):
        """ Switch to eval mode -> no parameter optimization """
        self.radialBasis.eval()
        if self.angularBasis is not None:
            self.angularBasis.eval()


    def train(self):
        """ Switch to train mode -> no parameter optimization """
        self.radialBasis.train()
        if self.angularBasis is not None:
            self.angularBasis.train()


    def computeDescriptors(self, mol):
        coords = np.asarray([mol.baseMol.xyz], dtype=np.float32)
        atTypes = np.asarray([mol.atNums], dtype=np.int64)
        res = self.computeDescriptorBatch(coords, atTypes)

        return res

    def computeDescriptorBatch(self, coords, aTypeList):
        """
          Compute the descriptors for one set of conformations witht he same number of atoms
          The conformations may differ in atom types.

          coords[nConf][nAtoms][xyz]
          aTypeList[nConf][nAtoms]

          returns tensor with descriptors [nConf][nDescriptors]
        """

        if self.angularBasis is not None:

            # lets try creating and first because they need more memory
            ang = self.angularBasis.computeDescriptorBatch(coords, aTypeList)
            rad = self.radialBasis.computeDescriptorBatch(coords, aTypeList)
            return torch.cat((rad,ang), dim=-1)

        return self.radialBasis.computeDescriptorBatch(coords, aTypeList)


    def nDescriptors(self):
        nDesc = self.radialBasis.nDescriptors()
        if self.angularBasis is not None:
            nDesc += self.angularBasis.nDescriptors()

        return nDesc
