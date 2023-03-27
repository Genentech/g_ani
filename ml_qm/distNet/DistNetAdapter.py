import copy
import logging
import os
from typing import Sequence, Dict, Tuple, List, Any, Union

import torch
import yaml
from torch import nn

from ml_qm import AtomInfo
from ml_qm.ANIMol import ANIMol
from ml_qm.distNet.data_set import DeviceDataSet, AtomTypes
from ml_qm.distNet.dist_net import EnergyNet
from t_opt.coordinates_batch import SameSizeCoordsBatch, CoordinateModelInterface
from t_opt.pytorch_computer import PytorchComputer
from t_opt.unit_conversion import Units, KCAL_MOL_to_AU, KCAL_MOL_A_to_AU

log = logging.getLogger(__name__)


class DistNetComputer(PytorchComputer):
    """
        Uses ml_qm package to calculate Neural Net Potentials for a batch of conformations.
        The conformations must have have the same atom count.
    """

    def __init__(self, nGPU:int, confFile:str, outputGrads:bool, compute_stdev:bool,
                 energyOutUnits:Units):
        """
        :param nGPU: Number of GPU to use
        :param confFile: configuration file for network yaml
        :param outputGrads: tif true gradients will be computed and reported
        :param compute_stdev: if true and the model is an ensample model the std_dev will be reported
        :param energyOutUnits: units for energy output
        """

        with open(confFile) as jFile:
            conf = yaml.load(jFile, Loader=yaml.FullLoader)

        device = torch.device("cuda" if nGPU > 0 and torch.cuda.is_available() else "cpu")
        log.info(f"device={device} nGPU={nGPU} cuda.isavaialble={torch.cuda.is_available()}") # pylint: disable=W1203
        confDir = os.path.dirname(confFile)
        initNet     = DistNetInitializer(conf,device, confDir)
        _model = initNet.create_coordinate_model(compute_stdev)
        _model.eval()

        atomTypes = []
        for at in conf['atom_types'].keys():
            atomTypes.append(AtomInfo.NameToNum[at])
        allowed_atom_num = frozenset(atomTypes)

        atomization_energies = torch.zeros((max(atomTypes)+1,),dtype=torch.float32)
        for at in atomTypes:
            atomization_energies[at] = ANIMol.atomEnergies[at]
        atomization_energies = atomization_energies.to(device=device)

        if energyOutUnits is Units.KCAL:
            eFactor = 1.
            fFactor = 1.
        elif energyOutUnits is Units.AU:
            eFactor = KCAL_MOL_to_AU   # if KCAL requested convert after calling NNP
            fFactor = KCAL_MOL_A_to_AU
            atomization_energies *= KCAL_MOL_to_AU
        else:
            raise TypeError("Unknown energy unit: {}".format(energyOutUnits))

        batch_by_atom_order = False  # atoms may be in random order
        mem_gb   = conf['mem_gb']
        memParam = conf['memoryParam']

        super().__init__(_model, allowed_atom_num, atomization_energies,
                         outputGrads, compute_stdev, torch.float32,
                         mem_gb, memParam, batch_by_atom_order,
                         eFactor, fFactor, energyOutUnits)


    def maxConfsPerBatch(self, nAtom:int) -> int:
        """
        :param nAtom: given a number of atoms compute how many conformations can be processed in one batch on the GPU

        May use self.MEM_GB and self.MEM_PARAM

        Overwrite this to compute more accurate max number of conforamtion based on atom count
        """

        param = self.MEM_PARAM
        if param is None: return super().maxConfsPerBatch(nAtom)

        # mem needed =  const + nConf * ( perConf + nAt * nA_x_nC + nAt^2 * nA2_x_nC + nAt^3 * nA2_x_nC )
        atMem = nAtom * ( param['nA_x_nC'] + nAtom * (param['nA2_x_nC'] + nAtom * param['nA3_x_nC']))

        return max(1,int((self.MEM_GB * 1024 * 1024 * 1024 - param['const']) / ( param['perConf'] + atMem )))


class CoordinateModel(CoordinateModelInterface):
    """
    the forward() method takes a SameSizeCoordinate batch as input and returns
    either a tensor of predicted energies or a tensor(Energies), tensor(stdev)
    depending on return_stdev
    """

    def __init__(self, atom_type_desc: AtomTypes, nnp_nets: Union[EnergyNet,List[EnergyNet]],
                 radial_cutoff:float, ang_cutoff:float, return_stdev=False):
        """
        Parameter
        ----------
        descriptor_module: module to compute descriptors from SameSizeCoordsBatch
        nnp_nets: single, or list of ANINet
        :param radial_cutoff: cutoff radius on radial descriptors
        :param ang_cutoff: cutoff radius on angular descriptors
        """

        super(CoordinateModel, self).__init__()

        self.radial_cutoff  = radial_cutoff
        self.ang_cutoff     = ang_cutoff
        self.atom_type_desc = atom_type_desc

        if not isinstance(nnp_nets, list): nnp_nets = [nnp_nets]
        if len(nnp_nets) == 1:
            if return_stdev:
                raise ValueError("return_stdev not supported for single nnp")

        self.ensemble_net = EnsembleNet(nnp_nets, return_stdev)


    def forward(self, inpt:SameSizeCoordsBatch):
        device = inpt.coords.device
        nConf = inpt.n_confs
        nAt   = inpt.n_atom_per_conf

        conf_info = torch.tensor([0.,nAt],device=device).repeat(nConf).reshape(nConf,-1)
        cidx = torch.arange(nConf)

        # col1 = conf number (1 per atom), col2 atom type
        atom_xyz = inpt.coords.reshape(-1,3)
        atom_info = torch.stack((cidx.repeat_interleave(nAt),inpt.atom_types.flatten()),-1)
        devDataSet = DeviceDataSet(self.atom_type_desc, conf_info, atom_info, atom_xyz)

        i_idx = torch.arange(nAt * nConf, dtype=torch.long).repeat_interleave(nAt)
        j_idx = torch.arange(nAt, dtype=torch.long).repeat(nAt).repeat(nConf).reshape(nConf, nAt * nAt)
        j_idx = (j_idx + nAt * torch.arange(nConf).unsqueeze(1)).flatten()
        ij_idx =  torch.cat((i_idx.unsqueeze(-1),j_idx.unsqueeze(-1)),1)

        dist_map:Dict[str, torch.tensor] = {}
        xyz = inpt.coords.reshape(nConf,nAt,3)
        dist = torch.cdist(xyz,xyz)

        fltr = (dist < self.radial_cutoff) & (dist > 0)

        # to make sure we have at least one descriptor for each atom we retain
        # the diagonal element for atoms with out neighbors
        # we will also set the distance to self.radial_cutoff in those cases to make sure the descs are 0
        noNeighbors = (~fltr.max(dim=-1)[0]).diag_embed()
        fltr = fltr | noNeighbors
        dist = dist + noNeighbors * self.radial_cutoff

        dist = dist[fltr]
        ij_idx = ij_idx[fltr.flatten()]
        dist_map['batch_atom_ij_idx'] =  ij_idx # tensor[ij.size,2]
        dist_map['batch_dist_ij'] =      dist    # tensor[ij.size]

        netIn: Dict[str, torch.tensor] = { }
        netIn['batch_dist_map'] = dist_map


        # TODO reuse from above later to improve performance but for now this is clearer
        # Above mith have indexes that are not quite correct
        i_idx = torch.arange(nAt * nConf, dtype=torch.long).repeat_interleave(nAt)
        j_idx = torch.arange(nAt, dtype=torch.long).repeat(nAt).repeat(nConf).reshape(nConf, nAt * nAt)
        j_idx = (j_idx + nAt * torch.arange(nConf).unsqueeze(1)).flatten()

        xyz = inpt.coords.reshape(nConf,nAt,3)
        dist = torch.cdist(xyz,xyz)
        fltr = (dist < self.ang_cutoff) & (dist > 0)
        fltr[fltr.sum(-1)<2,:] = False   # remove atoms with less than tw neighbors
        if fltr.sum() > 4:      # check that there is at least one atom with two neighbors
            dist = dist[fltr]
            fltr = fltr.flatten()
            i_idx=i_idx[fltr]
            j_idx=j_idx[fltr]

            # get counts and sortIndex such that atoms with same count are consecutive
            _, countNeigh = torch.unique_consecutive(i_idx, return_counts=True, dim=0)
            sortedCounts, _ = countNeigh.sort()
            countNeigh = countNeigh.repeat_interleave(countNeigh)
            _, sortIndex = countNeigh.sort() ### this needs to be stable, we need newest pytorch: https://github.com/pytorch/pytorch/issues/28871https://github.com/pytorch/pytorch/issues/28871

            # resort such that atoms with same count are consecutive
            dist = dist[sortIndex]
            i_idx = i_idx[sortIndex]
            j_idx = j_idx[sortIndex]

            # compute split sizes so we can split atoms with same neighbor count together
            countNeigh, countCounts = sortedCounts.unique_consecutive(return_counts=True)
            splitSizes = countNeigh * countCounts
            splitSizes = splitSizes.tolist()

            # split by neighbor count and reshape acocrting to neighbor count
            ang_neigh_map: Dict[int, Tuple[torch.tensor,torch.tensor,torch.tensor]] = {}
            dist_list = list(dist.split(splitSizes))
            i_idx_list = list(i_idx.split(splitSizes))
            j_idx_list = list(j_idx.split(splitSizes))
            for i, (countAt, nNeigh) in enumerate(zip(countCounts.tolist(),countNeigh.tolist())):
                ang_neigh_map[nNeigh] = (i_idx_list[i].reshape(countAt,1,-1)[:,:,-1],
                                         j_idx_list[i].reshape(countAt, -1),
                                         dist_list[i].reshape(countAt, -1))
            netIn['batch_ang_neighbor_map'] = ang_neigh_map

        netIn = self.ensemble_net(devDataSet, netIn)

        if isinstance(netIn, tuple): return netIn

        return netIn, None  # no stddev


class DistNetInitializer:
    """ Provides methods to instantiate an ani_net according to a configuration
        conf e.g. created from JSON as in train.json
    """

    def __init__(self, conf:Dict[str, Any], device, confDir = None):
        self.conf = conf
        self.device = device
        self.confDir = confDir
        self.atom_types = AtomTypes(conf['atom_types'], conf['atom_embedding'])
        self.conf = conf

    def create_coordinate_model(self, return_stdev=False):
        """ Create a CoordinateModel that can go straight from
            coordinates to energies and can be an ensamble model """

        nnp_models = self._create_models(self.conf['networkFiles'])
        if len(nnp_models) == 1 and return_stdev:
            log.warning("return_stdev not supported for single nnp, switched off")
            return_stdev = False

        radial_cutof:float = self.conf['radialNet']['radialCutoff']
        ang_cutof:float = self.conf['angleNet']['angularCutoff']
        cm = CoordinateModel(self.atom_types, nnp_models, radial_cutof, ang_cutof, return_stdev)
        return cm


    def _create_models(self, model_files:Sequence[str])->List[EnergyNet]:
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

            model = self._create_single_model(lConf, nnpFile)
            nnp_models.append(model)
        return nnp_models


    def _create_single_model(self, conf, modelFile)-> EnergyNet:
        """ returns the model according to the configuration

            If loadSavedDescriptorParam=false the parameters of the basis functions are
            taken as defined in the config file instead of from the checkpoint.
        """
        model = EnergyNet(conf)

        if modelFile is not None:
            log.info("loading model from: %s" % modelFile)
            checkpoint = torch.load(modelFile, map_location='cpu')
            model.load_state_dict(checkpoint['model'], False)

            model = model.to(self.device)

        return model

    #
    # def create_model(self, modelFiles, return_stdev=False):
    #     """ create a nnp model
    #
    #     Arguments
    #     ---------
    #     modelFiles: name of .nnp files. if None or string: a single model will be returned
    #                                     else: an ensemble model is returned
    #
    #     return_stdev: if True: prediction return value will be energies and stddev
    #     """
    #
    #     if modelFiles is not None and type(modelFiles) is not str and len(modelFiles) > 1:
    #         return self._create_ensemble_model(modelFiles, return_stdev)
    #
    #     if return_stdev:
    #         raise AttributeError("stdev requires multiple model weight files")
    #
    #     if modelFiles is not None and len(modelFiles) == 1:
    #         return self._create_single_model(self.conf, modelFiles[0])
    #
    #     return self._create_single_model(self.conf, modelFiles)
    #
    #
    # def _create_ensemble_model(self, model_files, return_stdev=False): # -> EnsembleNet:
    #     conf = self.conf
    #     nnp_models = []
    #     for mf in model_files:
    #         lConf = conf
    #         nnpFile = mf
    #         if isinstance(mf,dict):
    #             # replace config entries with network specific settings
    #             nnpFile = mf['file']
    #             lConf = copy.deepcopy(conf)
    #             for k, item in mf.items():
    #                 if k != "file": lConf[k] = item
    #
    #         nnpFile = os.path.join(self.confDir,nnpFile)
    #
    #         model = self._create_single_model(lConf, nnpFile, return_stdev)
    #         nnp_models.append(model)
    #
    #     return EnsembleNet(nnp_models)


class EnsembleNet(nn.Module):
    def __init__(self, nnp_nets:List[EnergyNet], return_stdev=False):
        super(EnsembleNet, self).__init__()

        self.nnp_nets = nnp_nets
        self.nEnsemble = len(nnp_nets)
        self.return_stdev = return_stdev

        for i, net in enumerate(nnp_nets):
            self.add_module("%s EnergyNEt" % i, net )


    def forward(self, devDataSet:DeviceDataSet, inp:Dict[str,torch.tensor]) -> Tuple[torch.tensor,torch.tensor]:
        finalEs = torch.empty((self.nEnsemble, devDataSet.n_confs),
                              dtype=torch.float32, device=devDataSet.device )

        for i, net in enumerate(self.nnp_nets):
            res = net.forward(devDataSet, inp)
            conf_idx = res['batch_output_conf_idx']
            e = res['batch_output']
            finalEs[i] = e[ conf_idx.argsort() ]

        if self.return_stdev:
            e = finalEs.detach()
            return finalEs.mean(0), e.std(0)
        return finalEs.mean(0), None

