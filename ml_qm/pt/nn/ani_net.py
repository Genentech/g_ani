import logging
import ml_qm.pt.nn.Precision_Util as pu
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611
from ml_qm.pt import ANIDComputer as AComput, RadialBasis
import ml_qm.pt.ThreeDistSqrBasis as ThreeDistSqrBasis
import ml_qm.pt.ThreeDistBasis as ThreeDistBasis
import ml_qm.pt.AngularBasis as AngularBasis
from ml_qm import AtomInfo
import copy
import os
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.dropout import Dropout
import pprint
from typing import Sequence, Tuple
from t_opt.coordinates_batch import SameSizeCoordsBatch,\
    CoordinateModelInterface
from ml_qm.nn.util import create_activation
pp = pprint.PrettyPrinter(indent=2, width=2000)





log = logging.getLogger(__name__)


def printWeights(model):
    for name, weights in model.named_parameters():
        warn("%s: %s" % (name,weights))
    warn("-"*30)



class AniAtomNet(nn.Module):

    """
       A fully connected NN computing the atomic energy contribution of atoms of type
       centerAtomNum.

       centerAtomNum -- atomic number of atom in questions
       nInputs -- number of input and hidden layer neurons
       nOutputs -- number of outpurts (generaly 1 = energy
       nLayers -- number of layers (last layer will have only one neuran)
       activation -- activation function
    """



    def __init__(self, centerAtomNum, nInputs, nOutputs, layerOpts,
                       defaultBias=True, hasOutBias=True, defaultInitWeight = {"method": "uniform", "gain": 1 },
                       defaultActivation=nn.Softplus(),
                       defaultBatchNormParam=None, defaultBatchNormBeforeActivation=False,
                       defaultDropOutPct=0):
        super(AniAtomNet, self).__init__()

        self.defaultLayerType='linear'
        self.defaultBias = defaultBias
        self.defaultInitWeight = defaultInitWeight
        self.defaultActivation = defaultActivation
        self.defaultBatchNormParam = defaultBatchNormParam
        self.defaultBatchNormBeforeActivation = defaultBatchNormBeforeActivation
        self.defaultDropOutPct = defaultDropOutPct

        self.mdl = nn.ModuleDict()

        numNodes=nInputs
        # generate network
        for i, opts in enumerate(layerOpts):
            out_features = self.createLayer(i, centerAtomNum, numNodes, opts )

            numNodes = out_features

        name = "%s olayer" %(AtomInfo.NumToName[centerAtomNum])
        layer = nn.Linear(numNodes, nOutputs, bias=hasOutBias)
        self.initWeigths(layer, numNodes, nOutputs, defaultInitWeight)
        # slower than without: layer = torch.jit.trace(layer, torch.rand(numNodes))

        if hasOutBias: nn.init.constant_(layer.bias, 0)
        self.mdl.add_module(name, layer)


    def forward(self, descriptors):
        """
           descriptors -- tensor[nAtoms, nDescriptors] for all atoms
                          with atomType for one or more molecules
        """

        for lyr in self.mdl.values():
            if not isinstance(lyr, nn.Bilinear):
                descriptors = lyr.forward(descriptors)
            else:
                descriptors = lyr.forward(descriptors,descriptors)

        return descriptors


    def initWeigths(self, layer, nInputs, nOutputs, initWeight = {"method": "uniform", "gain": 1 }):
        meth = initWeight['method']
        gain = initWeight['gain']
        if meth == 'uniform':
            stdv = gain / math.sqrt(nInputs)
            nn.init.uniform_(layer.weight, -stdv, stdv)
        elif meth == 'normal':
            stdv = gain / math.sqrt(nInputs)
            nn.init.normal_(layer.weight, 0., stdv)
        elif meth == 'khan':
            stdv = (gain / (nInputs + nOutputs)) ** 0.5
            nn.init.normal_(layer.weight, 0., stdv)
        elif meth == 'xavier_uniform':
            nn.init.xavier_uniform_(layer.weight, gain=gain)
        elif meth == 'xavier_normal':
            nn.init.xavier_normal_(layer.weight, gain=gain)
        else:
            raise TypeError("Unknown weight initialization '%s'" % meth)



    def createLayer(self, layerNum, centerAtomNum, in_features, confOptions=None ):
        """
            confOptions can be an int specifying the output features of a dict overwriting the options.
        """
        layerType        = self.defaultLayerType
        bias             = self.defaultBias
        activation       = self.defaultActivation
        batchNormParam   = self.defaultBatchNormParam
        batchNormBeforeAct=self.defaultBatchNormBeforeActivation
        dropOutPct       = self.defaultDropOutPct
        initWeight       = self.defaultInitWeight
        requires_grad    = True

        if np.isscalar(confOptions):
            out_features=confOptions
        else:
            out_features = confOptions["out_features"]
            if "type"                 in confOptions: layerType        = confOptions["type"]
            if "hasBias"              in confOptions: bias             = confOptions["hasBias"]
            if "batchNormParam"       in confOptions: batchNormParam   = confOptions["batchNormParam"]
            if "initWeight"           in confOptions: initWeight       = confOptions["initWeight"]
            if "batchNormBeforeActivation" in confOptions: batchNormBeforeAct= confOptions["batchNormBeforeActivation"]
            if "dropOutPct"                in confOptions: dropOutPct        = confOptions["dropOutPct"]
            if "activation" in confOptions:
                activation = create_activation(confOptions['activation'],
                                               confOptions.get("activationParam", {}))
            requires_grad = confOptions.get("requires_grad", True)

        name = "%s layer%d" % (AtomInfo.NumToName[centerAtomNum], layerNum)
        if layerType == "linear":
            layer = nn.Linear(in_features, out_features, bias=bias)
        elif layerType == "bilinear":
            layer = nn.Bilinear(in_features, in_features, out_features, bias=bias)

        # Initialize weights and bias
        self.initWeigths(layer, in_features, out_features, initWeight)
        if bias:
            nn.init.constant_(layer.bias, 0)

        self.mdl.add_module(name, layer)

        # batchnorm
        if batchNormBeforeAct and batchNormParam is not None:
            name = "%s batchNorm%d" % (AtomInfo.NumToName[centerAtomNum], layerNum)
            bNorm = BatchNorm1d(out_features, *batchNormParam)
            self.mdl.add_module(name, bNorm)

        # activation
        name = "%s act%d" % (AtomInfo.NumToName[centerAtomNum], layerNum)
        self.mdl.add_module(name, activation)

        # batchnorm
        if not batchNormBeforeAct and batchNormParam is not None:
            name = "%s batchNorm%d" % (AtomInfo.NumToName[centerAtomNum], layerNum)
            bNorm = BatchNorm1d(out_features, *batchNormParam)
            self.mdl.add_module(name, bNorm)

        if dropOutPct > 0:
            name = "%s dropOut%d" % (AtomInfo.NumToName[centerAtomNum], layerNum)
            self.mdl.add_module(name, Dropout(dropOutPct / 100., False))


        layer.requires_grad_(requires_grad)

        return out_features





class ANINet(nn.Module):

    """
        This implementation expects input in a slightly modified fasion from the in memory
        data source.

        actual network containing one AniAtomNet per atomic number to be described.

    """

    def __init__(self, atomTypes, nInputs, nOutputs, layerOpts, activation=nn.Softplus(),
                 hasBias=True, hasOutBias=True, initWeight = { "method": "uniform", "gain": 2 },
                 batchNormParam=None, batchNormBeforeActivation=False, dropOutPct=0):
        """
           atomTypes: array of supported atomic numbers
           nOutputs: number of ouput nodes
           nLayers: number of layers including output
           layerwidth: list of with of each hidden layer (output excluded)
        """
        super(ANINet, self).__init__()

        self.atomNets = {}       # indexed by index of atomTypes
        self.atomTypes = atomTypes
        self.nInputs = nInputs
        for atNum in atomTypes:
            n = AniAtomNet(atNum, nInputs, nOutputs, layerOpts, hasBias, hasOutBias, initWeight,
                           activation, batchNormParam, batchNormBeforeActivation, dropOutPct )
            self.atomNets[atNum] = n
            self.add_module("%s AtNet" % AtomInfo.NumToName[atNum], n)


    def printWeights(self):
        printWeights(self)


    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, confBatchDescription):

        device = confBatchDescription.atomDiscriptorList[0].device
        finalEs = torch.zeros(confBatchDescription.nConfs,
                              dtype=pu.NNP_PRECISION.NNPDType, device=device )

        # loop atom types in our batch
        for atType, atDesc, at2ConfNums in zip(confBatchDescription.uAtomTypes,
                          confBatchDescription.atomDiscriptorList,
                          confBatchDescription.at2confNumList):
            atNet = self.atomNets[atType]
            atEnergies = atNet.forward(atDesc)
            atEnergies = atEnergies.reshape(-1)

            # index is needed because some molecules might have no atoms of given type
            finalEs = pu.NNP_PRECISION.indexAdd_(finalEs, 0,at2ConfNums, atEnergies)

        return finalEs.to(dtype=pu.NNP_PRECISION.NNPDType)



    def forwardAtEnergies(self, confBatchDescription):
        """
        600-1000 seconds, clearly IO bound
        """
        device = confBatchDescription.atomDiscriptorList[0].device
        finalEs = torch.zeros(confBatchDescription.nConfs,
                              dtype=pu.NNP_PRECISION.indexAddDType, device=device )
        atEnergyMap = {}
        # loop over batch molecules

        for atType, atDesc, at2ConfNums in zip(confBatchDescription.uAtomTypes,
                          confBatchDescription.atomDiscriptorList,
                          confBatchDescription.at2confNumList):
            atNet = self.atomNets[atType]
            atEnergies = atNet.forward(atDesc)
            atEnergies = atEnergies.reshape(-1)

            atEnergyMap[atType] = atEnergies.clone()

            # index is needed because some molecules might have no atoms of given type
            finalEs = pu.NNP_PRECISION.indexAdd_(finalEs, 0,at2ConfNums, atEnergies)

        return finalEs.to(dtype=pu.NNP_PRECISION.NNPDType), atEnergyMap



class EnsembleNet(CoordinateModelInterface):
    def __init__(self, nnp_nets, return_stdev=False):
        super(EnsembleNet, self).__init__()

        self.nnp_nets = nnp_nets
        self.nEnsemble = len(nnp_nets)
        self.return_stdev = return_stdev

        for i, net in enumerate(nnp_nets):
            self.add_module("%s ANINet" % i, net )


    def forward(self, confBatchDescription) -> Tuple[torch.tensor,torch.tensor]:
        device = confBatchDescription.atomDiscriptorList[0].device
        finalEs = torch.empty((self.nEnsemble, confBatchDescription.nConfs),
                              dtype=pu.NNP_PRECISION.NNPDType, device=device )

        for i, net in enumerate(self.nnp_nets):
            finalEs[i] = net.forward(confBatchDescription)

        if self.return_stdev:
            e = finalEs.detach()
            return finalEs.mean(0), e.std(0)
        return finalEs.mean(0), None


class ByAtomTypeConfBatch():
    """
            Batch of conformational descriptors, already separated by unique atom type
            ready to go into atom_nets.
    """

    def __init__(self, nConfs, uAtomTypes, atomDescritorsList, at2confNumList, coords):
        """
            Batch of conformational descriptors, already separated by unique atom type
            ready to go into atom_nets.

            uAtomTypes array of unique atom types e.g. 1,6,7,8
            atomDescritorsList list of atomicDescriptorTensors one list element per uAtomType
               each element tensor with one entry per atom with given atom type, and nBasis columns
            at2confNumList list of conformer numbers for each atom

            e.g. if the batch contains one confs for each NH3 and HCN you will have:

                                       3 + 1 Hydrogen      1 C              2 N
               uAtomTypes =            1,                   6               7
               atomDiscriptorList = [ tensor[4,nBases] , tensor[1,nBasis, tensor[2,nbasis] ]
               at2confNumList     = [ [0,0,0,1]            [1]             [0,1] ]

            coords: coordinates of atoms (nConf,nAtomsPerConf,3)
        """


        self.nConfs             = nConfs
        self.uAtomTypes         = uAtomTypes
        self.atomDiscriptorList = atomDescritorsList
        self.at2confNumList     = at2confNumList
        self.coords             = coords


class DescriptorModule(nn.Module):
    def __init__(self, descriptor_computer):
        super(DescriptorModule, self).__init__()

        self.descriptor_computer = descriptor_computer


    def forward(self, same_size_coords_batch):
        """ Compute the Atomic Environment Descriptors and return a TorchConfBatchDescription
            which is pre-grouped by atom type.

            Arguments
            ---------
            same_size_coords_batch : tensor(n_confs,atPerConf) batch of coordinates fro conformations with atPerConf atoms
        """

        n_confs           = same_size_coords_batch.n_confs
        n_atom_per_conf   = same_size_coords_batch.n_atom_per_conf
        coords            = same_size_coords_batch.coords
        atom_types        = same_size_coords_batch.atom_types
        uniq_at_types     = same_size_coords_batch.uniq_at_types
        at_type_count     = same_size_coords_batch.at_type_count
        n_descriptor = self.descriptor_computer.nDescriptors()


        #######################################################################
        desc = self.descriptor_computer.computeDescriptorBatch(coords, atom_types)
        #######################################################################
        desc = desc.reshape(-1, n_descriptor)
        at2AtType = atom_types.reshape(-1)
        at2ConfNum = torch.arange(0,n_confs,1,dtype=torch.int64, device=desc.device)
        at2ConfNum = at2ConfNum.unsqueeze(-1).repeat(1,n_atom_per_conf).reshape(-1)

        #sort by atType
        at2AtType, sortByTypeUIdx = at2AtType.reshape(-1).sort()

        desc       = desc[sortByTypeUIdx]
        at2ConfNum = at2ConfNum[sortByTypeUIdx]

        #split by atom type
        descByAtTypeList    = desc.split(at_type_count)       # per atom type 1,6,7,8 descriptors per atom
        at2ConfNumList = at2ConfNum.split(at_type_count) # per atom type 1,6,7,8 reference from atom position to conformer number

        # atoms in coordsList are still sorted by original atom order in conf
        cbd = ByAtomTypeConfBatch(n_confs, uniq_at_types, descByAtTypeList, at2ConfNumList, coords)

        del desc, at2ConfNum, at2AtType
        return cbd


class CoordinateModel(nn.Sequential):
    """
    the forward() method takes a SameSizeCoordinate batch as input and returns
    either a tensor of predicted energies or a tensor(Energies), tensor(stdev)
    depending on return_stdev
    """

    def __init__(self, descriptor_module, nnp_nets, return_stdev=False):
        """
        Parameter
        ----------
        descriptor_module: module to compute descriptors from SameSizeCoordsBatch
        nnp_nets: single, or list of ANINet
        """

        super(CoordinateModel, self).__init__()

        self.desc_module = descriptor_module

        if not isinstance(nnp_nets, list): nnp_nets = [nnp_nets]
        if len(nnp_nets) == 1:
            if return_stdev:
                raise ValueError("return_stdev not supported for single nnp")

        self.energyNet = EnsembleNet(nnp_nets, return_stdev)


    def forward(self, inpt:SameSizeCoordsBatch):
        inpt = self.desc_module(inpt)
        inpt = self.energyNet(inpt)

        if isinstance(inpt, tuple): return inpt

        return inpt, None  # no stddev


class ANINetInitializer():
    """ Provides methods to instantiate an ani_net according to a configuration
        conf e.g. created from JSON as in train.json
    """

    def __init__(self, conf, device, confDir = None):
        pu.INIT_NNP_PrecisionUtil(**conf['Precision'])
        self.conf = conf
        self.device = device
        self.confDir = confDir


    def create_coordinate_model(self, return_stdev=False):
        """ Create a CoordinateModel that can go straight from
            coordinates to energies and can be an ensamble model """

        d_comptr = self.create_descriptor_computer()
        d_module = DescriptorModule(d_comptr)

        nnp_models = self._create_models(d_comptr, self.conf['networkFiles'])
        if len(nnp_models) == 1 and return_stdev:
            log.warning("return_stdev not supported for single nnp, switched off")
            return_stdev = False

        cm = CoordinateModel(d_module, nnp_models, return_stdev)
        return cm


    def _create_models(self, d_comptr:AComput.ANIDComputer, model_files:Sequence[str]):
        """ Create a list of models according to the configuration """

        conf = self.conf
        nnp_models = []
        for mf in model_files:
            lConf = conf
            nnpFile = mf
            if isinstance(mf,dict):
                # replace config entries with network specific settings
                nnpFile = mf['file']
                lConf = copy.deepcopy(conf)
                for k, item in mf.items():
                    if k != "file": lConf[k] = item

            nnpFile = os.path.join(self.confDir,nnpFile)

            model = self._create_single_model(lConf, nnpFile, d_comptr)
            nnp_models.append(model)

        return nnp_models


    def create_descriptor_computer(self):
        """ returns the model according to the configuration and the dscriptorComputer
            conf is a configuration object e.g. created from JSON as in train.json
        """

        atomTypes = []
        for at in self.conf['atomTypes']:
            atomTypes.append(AtomInfo.NameToNum[at])


        rbasis = getattr(RadialBasis, self.conf['radialBasis'], None)
        if rbasis is not None:
            rbasis = rbasis(atomTypes, **self.conf['radialBasisParam'], device=self.device)
        else:
            raise TypeError("unknown radialBasis: " + self.conf['radialBasis'])

        abasis = self.conf.get('angularBasis',None)
        if abasis is not None:
            abasis = getattr(AngularBasis, abasis, None)
            if abasis is not None:
                abasis = abasis(atomTypes, **self.conf['angularBasisParam'], device=self.device)
            else:
                abasis = getattr(ThreeDistBasis, self.conf['angularBasis'], None)
                if abasis is not None:
                    abasis = abasis(atomTypes, **self.conf['angularBasisParam'], device=self.device)
                else:
                    abasis = getattr(ThreeDistSqrBasis, self.conf['angularBasis'], None)
                    if abasis is not None:
                        abasis = abasis(atomTypes, **self.conf['angularBasisParam'], device=self.device)

                    elif self.conf['angularBasis'] != "None":
                        raise TypeError("unknown angularBasis: " + self.conf['angularBasis'], device=self.device)

        return AComput.ANIDComputer(atomTypes, rbasis, abasis)


    def create_model(self, modelFiles, desc_comput, loadSavedDescriptorParam=True, return_stdev=False):
        """ create a nnp model

        Arguments
        ---------
        modelFiles: name of .nnp files. if None or string: a single model will be returned
                                        else: an ensemble model is returned

        return_stdev: if True: prediction return value will be energies and stddev
        """

        if modelFiles is not None and type(modelFiles) is not str and len(modelFiles) > 1:
            return self._create_ensemble_model(modelFiles, desc_comput, loadSavedDescriptorParam, return_stdev)

        if return_stdev:
            raise AttributeError("stdev requires multiple model weight files")

        if modelFiles is not None and len(modelFiles) == 1:
            return self._create_single_model(self.conf, modelFiles[0], desc_comput, loadSavedDescriptorParam)

        return self._create_single_model(self.conf, modelFiles, desc_comput, loadSavedDescriptorParam)



    def _create_single_model(self, conf, modelFile, descComput, loadSavedDescriptorParam=True):
        """ returns the model according to the configuration

            If loadSavedDescriptorParam=false the parameters of the basis functions are
            taken as defined in the config file instead of from the checkpoint.
        """
        atomTypes = []
        for at in conf['atomTypes']:
            atomTypes.append(AtomInfo.NameToNum[at])

        actFun = create_activation(conf['activation'],
                                   conf.get("activationParam", {}))


        batchNormBeforActivation = conf.get('batchNormBeforeActivation', False)
        batchNormParam = conf.get('batchNormParam', None)
        if batchNormParam == "None": batchNormParam = None

        nInput = descComput.nDescriptors()
        hasOutBias=conf.get('hasBias.outLayer', True)
        defInitWeight = conf.get('initWeight', { "method": "uniform", "gain": 2 })
        model = ANINet(atomTypes, nInput, nOutputs=1, layerOpts=conf['layers'],
                       activation=actFun, hasBias=conf['hasBias'], hasOutBias=hasOutBias,
                       initWeight = defInitWeight,
                       batchNormParam=batchNormParam, batchNormBeforeActivation=batchNormBeforActivation,
                       dropOutPct=conf['dropOutPct'] )

        if modelFile is not None:
            log.info("loading model from: %s" % modelFile)
            checkpoint = torch.load(modelFile, map_location='cpu')
            model.load_state_dict(checkpoint['model'], False)
            if loadSavedDescriptorParam: descComput.load_state_dict(checkpoint['descComputer'])

            model = model.to(self.device)
            descComput.to(self.device)

        model = model.type(pu.NNP_PRECISION.NNPDType)
        return model


    def _create_ensemble_model(self, model_files, desc_comput, loadSavedDescriptorParam=True, return_stdev=False):

        if not loadSavedDescriptorParam :
            raise TypeError("Creating an ensemble model without loading the descriptor parameters does not make sense")

        conf = self.conf
        nnp_models = []
        for mf in model_files:
            lConf = conf
            nnpFile = mf
            if isinstance(mf,dict):
                # replace config entries with network specific settings
                nnpFile = mf['file']
                lConf = copy.deepcopy(conf)
                for k, item in mf.items():
                    if k != "file": lConf[k] = item

            nnpFile = os.path.join(self.confDir,nnpFile)

            model = self._create_single_model(lConf, nnpFile, desc_comput, return_stdev)
            nnp_models.append(model)

        return EnsembleNet(nnp_models)
