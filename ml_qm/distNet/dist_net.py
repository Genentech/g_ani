"""
Created on Jul 30, 2019

@author: albertgo
"""
import logging

import torch
import torch.nn as nn
from typing import Dict, Tuple, Any, Union, List
import math
import copy

from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.dropout import Dropout
from cdd_chem.util.io import warn  # noqa: F401; # pylint: disable=W0611
from ml_qm.distNet.batch_data_set import BatchDataSet
from ml_qm.distNet.data_set import DeviceDataSet
from ml_qm.pt.nn.Precision_Util import NNP_PRECISION
from ml_qm.nn.util import initBias, initWeigths, create_activation


log = logging.getLogger(__name__)
torch.set_printoptions(precision=2, threshold=9999, linewidth=9999, sci_mode=False)
PROFILE = 0


class ParallelEnergyNet(nn.Module):
    def __init__(self, conf:Dict[str,Any], data_set:BatchDataSet):
        super().__init__()

        self.energy_net = EnergyNet(conf)


    def forward(self, ds:BatchDataSet, batch_idx:torch.tensor):
        if batch_idx.shape[0] == 0: return
        assert batch_idx.shape[0] == 1, f"This should only be used with DataParrallel {batch_idx}"
        batch = ds.batches[batch_idx]

        device_dat_set = ds.get_device_data_set(batch_idx.device)

        return self.energy_net(device_dat_set, batch)


    def load_state_dict(self, state_dict, strict=True):
        """ Hide ParallelEnergyNet, it should not contain state """
        self.energy_net.load_state(state_dict, strict)


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """ Hide ParallelEnergyNet, it should not contain state """
        return self.energy_net.state_dict(destination, prefix, keep_vars)


    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnergyNet(nn.Module):
    """ Energy Net is the main Network that includes teh networks to compute the radial and angular descriptors
        as well as the linear layers that compute the enrgies fromt he aggregated values.
    """
    def __init__(self, conf:Dict[str,Any]):
        super(EnergyNet, self).__init__()

        atom_embedding_size = len(list(conf['atom_types'].values())[0])
        self.computInput = ComputeDescriptors(conf)

        conf = conf['energyNet']
        self.add_center_atom_embedding = conf.get('addCenterAtomEmbedding', True)

        self.mod_dict = nn.ModuleDict()

        nInputs = self.computInput.n_output

        # we are reading the atomic embedding of for the center atom
        if self.add_center_atom_embedding:
            nInputs += atom_embedding_size

        for layer_num, layer_conf in enumerate(conf['layers']):
            nOut       = layer_conf['nodes']
            bias       = layer_conf['bias']
            activation = layer_conf.get('activation', None)
            batchNorm  = layer_conf.get('batchNorm', None)
            dropOutPct = layer_conf.get('dropOutPct', 0)
            requires_grad = layer_conf.get('requires_grad', True)

            layer = nn.Linear(nInputs, nOut, bias=bias)
            initWeigths(layer, nInputs, nOut, layer_conf.get('initWeight', None))
            if bias: initBias(layer, layer_conf.get('initBias', None))
            name = "EnergyNet %d (%d)" % (layer_num, nOut)
            self.mod_dict.add_module(name,layer)

            # activation
            if activation is not None:
                activation = create_activation(
                                activation['type'], activation.get('param',{}))
                name = "EnergyNet act%d" % layer_num
                self.mod_dict.add_module(name, activation)

            if  batchNorm is not None:
                name = "EnergyNet batchNorm%d" % layer_num
                bNorm = BatchNorm1d(nOut, *batchNorm.get('param',[]))
                self.mod_dict.add_module(name, bNorm)

            if dropOutPct > 0:
                name = "EnergyNet dropOut%d" % layer_num
                self.mod_dict.add_module(name, Dropout(dropOutPct / 100., False))

            layer.requires_grad_(requires_grad)
            nInputs = nOut

        self.n_output = nInputs


    def forward(self, data_set:Union[BatchDataSet,DeviceDataSet], inp:Dict[str,torch.tensor]) -> Dict[str,torch.tensor]:
        """
        :param data_set: BatchDataSet or DeviceDataSet
        :param inp: must contain 'batch_ang_neighbor_map', 'batch_dist_map'
        :return: dict eith 'batch_output': conformational energies , 'batch_output_conf_idx': indeces
        """
        if isinstance(data_set, BatchDataSet):
            data_set = data_set.get_device_data_set()  # get part that is on this device

        # shallow copy because inp will be re-used
        inp = self.computInput(data_set, copy.copy(inp))

        # batch_rad_center_atom_idx  = inp['batch_rad_center_atom_idx']
        desc = inp.pop('batch_desc')

        if self.add_center_atom_embedding:
            atom_idxs = inp['batch_center_atom_idx']
            desc = torch.cat((desc, data_set.atom_types.atom_embedding(data_set.atoms_long[atom_idxs,1])),dim=1)

        if PROFILE > 0:
            # record so trace is easier to follow in chrome://tracing
            torch.acos_(data_set.ZERO)
            torch.acos_(data_set.ZERO)
#             if 'cuda' in str(desc.device):
#                 torch.cuda.synchronize()

        for layr in self.mod_dict.values():
            desc = layr(desc)

        atomic_energies = desc.flatten()
        atom_idxs = inp.pop('batch_center_atom_idx')

        # now we need to add by conf_idx
        conf_idx = data_set.atoms_long[atom_idxs,0]
        conf_idx, atom_pos_to_conf_pos = conf_idx.unique(return_inverse=True)
        #warn(f'unique: {conf_idx.shape} {atom_pos_to_conf_pos.shape}')
        conf_energies = torch.zeros_like(conf_idx, dtype=desc.dtype)
        conf_energies.index_add_(0, atom_pos_to_conf_pos, atomic_energies)

        inp['batch_output']          = conf_energies
        inp['batch_output_conf_idx'] = conf_idx

        if PROFILE > 0:
            # record so trace is easier to follow in chorme://tracing
            torch.acos_(data_set.ZERO)
            torch.isinf(data_set.ZERO)
#             if 'cuda' in str(conf_energies.device):
#                 torch.cuda.synchronize()

        return inp


    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ComputeDescriptors(nn.Module):
    """
        combine radial and angular descriptors.
    """

    def __init__(self, conf:Dict[str,Any]):
        super(ComputeDescriptors, self).__init__()

        self.angle_net = None
        self.n_output = 0

        aNet = conf.get('angleNet', None)
        if aNet is not None:
            self.angle_net = AngleNet1(conf)
            self.n_output += self.angle_net.n_output

        self.rad_net  = RadialNet(conf)
        self.n_output += self.rad_net.n_output


    def forward(self, data_set:DeviceDataSet, inp:Dict[str, torch.tensor]) -> Dict[str,torch.tensor]:
        """ concatenate inputs from Radial and angular descriptors
        """
        if self.angle_net is None:
            # no angular desc just compute radial

            inp = self.rad_net(data_set, inp)

            desc_rad = inp.pop('batch_rad_desc')
            idx_rad = inp.pop('batch_rad_center_atom_idx')
            inp['batch_desc']            = desc_rad
            inp['batch_center_atom_idx'] = idx_rad

            return inp

        if 'batch_ang_neighbor_map' in inp: # some molecules might not have triples
            inp = self.angle_net(data_set, inp)

        inp = self.rad_net(data_set, inp)

        # if this is slow we might be able to implement this:
        #https://github.com/rusty1s/pytorch_sparse/blob/7bb2fac56edfa15d89e4d845ac31f72f8e7f6067/torch_sparse/coalesce.py

        idx_rad = inp.pop('batch_rad_center_atom_idx')
        idx_ang = inp.pop('batch_ang_center_atom_idx', None)

        if PROFILE > 0:
            # record so trace is easier to follow in chorme://tracing
            torch.acos_(data_set.ZERO)
            torch.asin_(data_set.ZERO)
#             if 'cuda' in str(idx_ang.device):
#                 torch.cuda.synchronize()

        if idx_ang is not None:
            idx_both = torch.cat((idx_ang,idx_rad))
            idx_both, inv_both = idx_both.unique(return_inverse=True)
            inv_ang = inv_both[0:idx_ang.shape[0]]
            inv_rad = inv_both[idx_ang.shape[0]:]
            del idx_ang, idx_rad

            desc_ang = inp.pop('batch_ang_desc')
            desc_rad = inp.pop('batch_rad_desc')

            desc = torch.zeros((idx_both.shape[0],self.n_output),
                               dtype=desc_rad.dtype, device=desc_rad.device)
            # place compute angular descriptors in left part of desc
            desc[inv_ang,0:desc_ang.shape[-1]] = desc_ang
            del inv_ang

            # place computed radial descriptors in right part of desc
            desc[inv_rad,desc_ang.shape[-1]:] = desc_rad
            del inv_rad, desc_ang, desc_rad
        else:
            # no atom had a triple because atoms are too far apart
            desc_rad = inp.pop('batch_rad_desc')
            desc = torch.zeros((idx_rad.shape[0], self.n_output),
                               dtype=desc_rad.dtype, device=desc_rad.device)

            idx_both = idx_rad
            # place computed radial descriptors in right part of desc
            desc[:, (self.n_output-desc_rad.shape[-1]):] = desc_rad
            del desc_rad

        inp['batch_desc']            = desc
        inp['batch_center_atom_idx'] = idx_both

        return inp

class RadialNet(nn.Module):
    def __init__(self, conf:Dict[str,Any]):
        super(RadialNet, self).__init__()

        self.computInput = ComputeRadialInput(conf)

        conf = conf['radialNet']

        self.mod_dict = nn.ModuleDict()
        self.cutoff = conf['radialCutoff']

        nInputs = self.computInput.n_output

        for layer_num, layer_conf in enumerate(conf['layers']):
            nOut       = layer_conf['nodes']
            bias       = layer_conf['bias']
            activation = layer_conf.get('activation', None)
            batchNorm  = layer_conf.get('batchNorm', None)
            dropOutPct = layer_conf.get('dropOutPct', 0)
            requires_grad = layer_conf.get('requires_grad', True)

            layer = nn.Linear(nInputs, nOut, bias=bias)
            initWeigths(layer, nInputs, nOut, layer_conf.get('initWeight', None))
            if bias: initBias(layer, layer_conf.get('initBias', None))
            name = "RadialNet %d (%d)" % (layer_num, nOut)
            self.mod_dict.add_module(name,layer)

            # activation
            if activation is not None:
                activation = create_activation(
                                activation['type'], activation.get('param',{}))
                name = "RadialNet act%d" % layer_num
                self.mod_dict.add_module(name, activation)

            if  batchNorm is not None:
                name = "RadialNet batchNorm%d" % layer_num
                bNorm = BatchNorm1d(nOut, *batchNorm.get('param',[]))
                self.mod_dict.add_module(name, bNorm)

            if dropOutPct > 0:
                name = "RadialNet dropOut%d" % layer_num
                self.mod_dict.add_module(name, Dropout(dropOutPct / 100., False))

            layer.requires_grad_(requires_grad)
            nInputs = nOut

#         self.add_module("RadialNetDict", self.mod_dict)
        self.n_output:int = nInputs

    def forward(self, data_set:DeviceDataSet, inp:Dict[str,torch.tensor]) -> Dict[str,torch.tensor]:

        # prepare input
        device = data_set.ZERO.device
        batch_dist_map = inp.pop('batch_dist_map')
        atom_ij_idx = batch_dist_map['batch_atom_ij_idx'].to(device, non_blocking=True)
        dist_ij = batch_dist_map['batch_dist_ij'].to(device, dtype=torch.float32, non_blocking=True)
        inp['batch_dist_ij']     = dist_ij
        inp['batch_atom_ij_idx'] = atom_ij_idx
        del atom_ij_idx, batch_dist_map

        inp = self.computInput(data_set, inp)

        desc = inp.pop('batch_rad_input')  # cat( pair_dist_ij,atom_embed_i, atom_embed_j)

        if PROFILE > 0:
            # record so trace is easier to follow in chorme://tracing
            torch.acos_(data_set.ZERO)
            torch.atan_(data_set.ZERO)
#             if 'cuda' in str(desc.device):
#                 torch.cuda.synchronize()


        # call the layer that compute the radial AEV
        for layr in self.mod_dict.values():
            desc = layr(desc)

        # now we need to multiply each with the cutoff function
        # we could try using a bump function here for performance?
        fc = (0.5 * torch.cos(math.pi * dist_ij / self.cutoff) + 0.5)
        #fc = (0.5 * torch.cos(math.pi * torch.clamp(in_desc[:,0],0,1)) + 0.5)
        desc = desc * fc.view(-1,1)

        # sum up all descriptors so we have one per center atom
        # todo:  faster alternative to index_add???

        batch_rad_center_atom_idx = inp['batch_rad_center_atom_idx']
        idx_rad, inv_rad = batch_rad_center_atom_idx.unique(return_inverse=True)
        desc_rad = torch.zeros((idx_rad.shape[0],desc.shape[-1]),
                           dtype=desc.dtype, device=desc.device)
        desc_rad.index_add_(0, inv_rad, desc)

        inp['batch_rad_desc'] = desc_rad
        inp['batch_rad_center_atom_idx'] =  idx_rad
        return inp


class VDWNormalizedReciprocalDistance(nn.Module):
    def __init__(self):
        super(VDWNormalizedReciprocalDistance, self).__init__()

    def forward(self, data_set:DeviceDataSet, inp:Dict[str,torch.tensor]) -> torch.tensor:
        """
            Replace first column in desc which is dist_ij with (vdw(i)+vdw(j))/(2* dist_ij)
        """
        atom_ij_idx = inp['batch_atom_ij_idx']
        atom_types  = data_set.atom_types
        atom_nums   = data_set.atoms_long[:,1]
        dist_ij     = inp['batch_dist_ij']

        dist_ij = atom_types.atom_vdw[atom_nums[atom_ij_idx]].sum(-1)/2/dist_ij

        inp['batch_dist_ij'] = dist_ij

        return inp


class ComputeRadialInput(nn.Module):
    """
        Adds the radial descriptors to the input dict as:
            inp['batch_rad_input'], inp['batch_rad_center']
        The new tensors contain the descriptors and the center atom idxs
        Note each center atom i can appear multiple time.
        Lets try to do the summation while combining using sparse tensors

        If that is slow we will have to carry a tensor
        with atom_i_position parallel to atom_i_idx and use index_add or scatter_add
    """

    def __init__(self, conf:Dict[str,Any]):
        super(ComputeRadialInput, self).__init__()
        self.cutoff = conf['radialNet']['radialCutoff']

        fuz = conf['radialNet'].get('fuzz',{'minrad': [], 'stddev': [] })
        self.fuz_min = fuz['minrad'] if isinstance(fuz['minrad'], list) else [fuz['minrad']]
        self.fuz_std = fuz['stddev'] if isinstance(fuz['stddev'], list) else [fuz['stddev']]

        n_feature = len(list(conf['atom_types'].values())[0])

        n_distcolumns = 1
        self.add_square = conf['radialNet'].get("addSquare", False)
        self.add_reciproce = conf['radialNet'].get("addReciproce", False)

        self.n_output = n_distcolumns + 2 * n_feature
        if self.add_square:    self.n_output += 1
        if self.add_reciproce: self.n_output += 1

        self.distanceNormalizer = None
        normalizeDistance = conf['radialNet'].get('normalizeDistance', None)
        if normalizeDistance == 'reciprocalVDW':
            self.distanceNormalizer = VDWNormalizedReciprocalDistance()


    def forward(self, ds:DeviceDataSet, inp:Dict[str,torch.tensor]) -> Dict[str,torch.tensor]:
        """ generate input for network that works on atom pairs
        """
        if self.distanceNormalizer:
            inp = self.distanceNormalizer(ds,inp)

        if PROFILE > 0:
            # record so trace is easier to follow in chrome://tracing
            torch.acos_(ds.ZERO)
            torch.cosh_(ds.ZERO)
#             if 'cuda' in str(atoms_xyz.device):
#                 torch.cuda.synchronize()

        atom_ij_idx = inp['batch_atom_ij_idx']
        dist_ij     = inp['batch_dist_ij']
        del inp['batch_dist_ij'], inp['batch_atom_ij_idx']

        # if len(dist_ij) == 0:
        #     inp['batch_rad_center_atom_idx'] = None
        #     inp['batch_rad_input'] = None
        #     return inp
        #
        neigh_atom_embed_ij = ds.atom_types.atom_embedding(ds.atoms_long[atom_ij_idx,1])

        dist_ij = dist_ij.unsqueeze(-1)
        if self.training:
            for fmin, fstd in zip(self.fuz_min, self.fuz_std):
                ## apply fuz only to distances larger than fuz_min and normal distributed by the amount dist's are larger than fuz_min
                minDist = torch.clamp(dist_ij - fmin, min=0.)
                dist_ij = dist_ij.addcmul(minDist,torch.randn_like(minDist), value=fstd)
                del minDist

        # only embedding of atom j [1] is divided by dist_ij
        jembed = neigh_atom_embed_ij[:,1,:].clone()
        jembed = jembed / dist_ij
        neigh_atom_embed_ij[:,1,:] = jembed

        rad_desc_list = [dist_ij, neigh_atom_embed_ij.reshape(dist_ij.shape[0],-1)]
        if self.add_square:    rad_desc_list.append(dist_ij*dist_ij/self.cutoff)
        if self.add_reciproce: rad_desc_list.append(self.cutoff/dist_ij)

        #dist_ij = dist_ij/self.cutoff
        del dist_ij
        rad_desc = torch.cat(rad_desc_list, dim=-1)

        inp['batch_rad_center_atom_idx'] = atom_ij_idx[:,0]
        inp['batch_rad_input']           = rad_desc
        return inp


class AngleNet(nn.Module):
    def __init__(self, conf:Dict[str,Any]):
        super(AngleNet, self).__init__()

        n_feature = len(list(conf['atom_types'].values())[0])
        fuz = conf['angleNet'].get('fuzz',{'minrad': [], 'stddev': [], 'angleStddev_deg': None })
        fuz_min = fuz['minrad'] if isinstance(fuz['minrad'], list) else [fuz['minrad']]
        fuz_std = fuz['stddev'] if isinstance(fuz['stddev'], list) else [fuz['stddev']]
        if fuz['angleStddev_deg'] is not None:
            raise RuntimeError("angleStddev_deg is only suypported for angle-cutoff and angle")
        self.computInput = ComputeAngleInput(n_feature, fuz_min, fuz_std)

        conf = conf['angleNet']

        self.mod_dict = nn.ModuleDict()
        self.cutoff = conf['angularCutoff']

        nInputs = self.computInput.n_output

        for layer_num, layer_conf in enumerate(conf['layers']):
            nOut       = layer_conf['nodes']
            bias       = layer_conf['bias']
            activation = layer_conf.get('activation', None)
            batchNorm  = layer_conf.get('batchNorm', None)
            dropOutPct = layer_conf.get('dropOutPct', 0)
            requires_grad = layer_conf.get('requires_grad', True)

            layer = nn.Linear(nInputs, nOut, bias=bias)
            initWeigths(layer, nInputs, nOut, layer_conf.get('initWeight', None))
            if bias: initBias(layer, layer_conf.get('initBias', None))
            name = "AngleNet %d (%d)" % (layer_num, nOut)
            self.mod_dict.add_module(name,layer)

            # activation
            if activation is not None:
                activation = create_activation(
                                activation['type'], activation.get('param',{}))
                name = "AngleNet act%d" % layer_num
                self.mod_dict.add_module(name, activation)

            if  batchNorm is not None:
                name = "AngleNet batchNorm%d" % layer_num
                bNorm = BatchNorm1d(nOut, *batchNorm.get('param',[]))
                self.mod_dict.add_module(name, bNorm)

            if dropOutPct > 0:
                name = "AngleNet dropOut%d" % layer_num
                self.mod_dict.add_module(name, Dropout(dropOutPct / 100., False))

            layer.requires_grad_(requires_grad)
            nInputs = nOut

#         self.add_module("AngleNetDict", self.mod_dict)
        self.n_output = nInputs


    def forward(self, data_set:DeviceDataSet, inp:Dict[str,torch.tensor]) -> Dict[str,torch.tensor]:
        inp = self.computInput(data_set, inp)
        n_feature = self.computInput.n_output

        if PROFILE > 0:
            # record so trace is easier to follow in chorme://tracing
            torch.acos_(data_set.ZERO)
            torch.erf_(data_set.ZERO)

        batch_ang_input_map = inp.pop('batch_ang_input_map')
        batch_ang_center_map  = inp.pop('batch_ang_center_map')

        desc_list = []
        for nNeigh, desc in batch_ang_input_map.items():
            in_desc = desc
            nCenter = desc.shape[0]
            nTriple = desc.shape[1]
            desc = desc.reshape(-1,n_feature) # flatten out ntriple dimension for batchnorm

            # we might need to reshape desc so that it has only 2d
            for lyr in self.mod_dict.values():
                desc = lyr(desc)

            desc = desc.reshape(nCenter, nTriple, -1)

                # now we need to multiply each with the cutoff function
                # we could try using a bump function here for performance?
            # in_desc[:,:,0:2] are the ij and ik distances
            fc = (0.5 * torch.cos(math.pi * torch.clamp(in_desc[:,:,0:2] / self.cutoff,0,1)) + 0.5)
            fc = fc.prod(dim=-1,keepdim=True)
            desc = desc * fc
            desc = NNP_PRECISION.sum( desc, 1 )
            #Does the deviation warnet convertign to double for the summation?
            #warn(f"ang sum diff: {(desc.sum(dim=1).double() - desc.double().sum(dim=1)).abs().max()*1000}")

            desc_list.append(desc)

        desc            = torch.cat(desc_list)
        center_atom_idx = torch.cat(tuple(batch_ang_center_map.values()))
        inp['batch_ang_desc'] = desc
        inp['batch_ang_center_atom_idx'] = center_atom_idx
        return inp


class ComputeAngleInput(nn.Module):
    """
        Adds the angular descriptors to the input dict as:
            inp['batch_ang_input_map']
            inp['batch_ang_center_map']
        The new maps are keyed by number of neighbors around the center atom

        dimensions of batch_ang_input_map items are:
           - nCenterAt number of center atoms i, with the same number of neighbors
           - nNeigh * (nNeigh -1): number or triples for this nNeigh
           - 3 (distances ij,ij,jk_norm) + 3 * n_Embedding_features: ijk atoms
        batch_ang_center_map items contain the atom indices of the center atoms i
           of each triple ijk
    """

    def __init__(self, n_feature:int, fuz_min:List[float] = [], fuz_std:List[float] = [], fuz_angStd:float = 0.):
        super(ComputeAngleInput, self).__init__()

        n_distcolumns = 3
        self.n_output = n_distcolumns + 3 * n_feature
        self.fuz_min = fuz_min
        self.fuz_std = fuz_std
        self.fuz_angStd = fuz_angStd


    def _computeNormalizedJKDistance(self, ds:DeviceDataSet, triple_j:torch.tensor, triple_k:torch.tensor,
                                           triple_dist_ij:torch.tensor, triple_dist_ik:torch.tensor):
        """
            return jk distance normalizes
        """
        atoms_xyz = ds.atoms_xyz
        triple_dist_jk = (atoms_xyz[triple_j] - atoms_xyz[triple_k]).norm(dim=-1)

        # normalize to be between 0 and 1 based on triangle equation:
        # jk distance must be between |dij-dik| and dij+dik
        maxDist = torch.max(triple_dist_ij, triple_dist_ik)
        minDist = torch.min(triple_dist_ij, triple_dist_ik)
        triple_dist_jk = (triple_dist_jk - maxDist + minDist) / (2 * minDist)

        return triple_dist_jk


    def _computeNormalizedJKDistanceSqr(self, ds:DeviceDataSet, triple_j:torch.tensor, triple_k:torch.tensor,
                                              triple_dist_ij:torch.tensor, triple_dist_ik:torch.tensor):
        """ (return jk distance)^2 normalized """
        atoms_xyz = ds.atoms_xyz
        triple_dist_sqr_jk = (atoms_xyz[triple_j] - atoms_xyz[triple_k]).pow(2).sum(dim=-1)

        # normalize to be between 0 and 1 based on triangle equation:
        # jk distance must be between (dij-dik)^2 and (dij+dik)^2
        triple_dist_sqr_jk = (triple_dist_sqr_jk - (triple_dist_ij - triple_dist_ik).pow(2)) / (4 * triple_dist_ij * triple_dist_ik)

        return triple_dist_sqr_jk


    def forward(self, ds:DeviceDataSet, inp:Dict[str,torch.tensor]) -> Dict[str,torch.tensor]:
        """ generate input for network that works on atom triples
        """

        if PROFILE > 0:
            # record so trace is easier to follow in chorme://tracing
            torch.acos_( ds.ZERO)
            torch.expm1_(ds.ZERO)

        batch_ang_center_map: Dict[int,torch.tensor] = {}
        batch_ang_input_map: Dict[int,torch.tensor] = {}
        inp['batch_ang_center_map'] = batch_ang_center_map
        inp['batch_ang_input_map'] = batch_ang_input_map
        batch_ang_neighbor_map = inp.pop('batch_ang_neighbor_map')

        for nNeigh, (atom_i_idx, atom_j_idx, dist_ij) in batch_ang_neighbor_map.items():
            if nNeigh == 1: continue

            nCenterAt = atom_i_idx.shape[0] # number of i atoms with nNeigh neighbors
            atom_i_idx= atom_i_idx.to(atom_i_idx.device, non_blocking=True)
            atom_j_idx= atom_j_idx.to(atom_i_idx.device, non_blocking=True)
            dist_ij   = dist_ij.to(atom_i_idx.device, non_blocking=True)

            if self.training:
                for fmin, fstd in zip(self.fuz_min, self.fuz_std):
                    if fstd != 0.:
                        ## apply fuz only to distances larger than fuz_min and normal distributed by the amount dist's are larger than fuz_min
                        minDist = torch.clamp(dist_ij - fmin, min=0.)
                        dist_ij = dist_ij.addcmul(minDist, torch.randn_like(minDist), value=fstd)

            triple_dist_ij, triple_dist_ik, triple_dist_jk \
                = self._computeTriples(ds, nNeigh, atom_i_idx, atom_j_idx, dist_ij)
            del dist_ij

            neigh_atom_embed_i = ds.atom_types.atom_embedding(ds.atoms_long[atom_i_idx,1])
            neigh_atom_embed_j = ds.atom_types.atom_embedding(ds.atoms_long[atom_j_idx,1])
            neigh_atom_embed_i, neigh_atom_embed_j, neigh_atom_embed_k \
               = torch.broadcast_tensors(
                    neigh_atom_embed_i.view(nCenterAt,1,1,-1),
                    neigh_atom_embed_j.view(nCenterAt,nNeigh,1,-1),
                    neigh_atom_embed_j.view(nCenterAt,1,nNeigh,-1))
            triple_dist_ij = triple_dist_ij.unsqueeze(-1)
            triple_dist_ik = triple_dist_ik.unsqueeze(-1)
            triple_dist_jk = triple_dist_jk.unsqueeze(-1)
            ang_desc = torch.cat((
                triple_dist_ij, triple_dist_ik, triple_dist_jk,
                neigh_atom_embed_i,
                neigh_atom_embed_j/triple_dist_ij, neigh_atom_embed_k/triple_dist_ik), dim=-1)

            # filter out diagonal elements in jk
            filtr = ~torch.eye(nNeigh,dtype=torch.uint8, device=ang_desc.device).bool()
            # below gets lower loss because it includes diagonal elements????
            #ang_desc = ang_desc.reshape(nCenterAt,nNeigh*nNeigh,-1)
            ang_desc = ang_desc[:,filtr,:]

            batch_ang_center_map[nNeigh] = atom_i_idx.view(-1)
            batch_ang_input_map[nNeigh]  = ang_desc

        return inp


    @staticmethod
    def debugAngDesc(batch_ang_center_map, batch_ang_input_map):
        nCounts = list(batch_ang_center_map.keys())
        nCounts.sort()

        atidx = []
        desc = []
        for nc in  nCounts:
            ai = batch_ang_center_map[nc].repeat_interleave(nc)
            atidx.append(ai)
            desc.append(batch_ang_input_map[nc].reshape(ai.shape[0],-1))
        atidx = torch.cat(atidx)
        desc  = torch.cat(desc)
        print(f"atidx: {atidx.shape} desc {desc.shape}")
        print(atidx)
        print(desc)


    def _computeTriples(self, data_set:DeviceDataSet, nNeigh:int, atom_i_idx:torch.tensor, atom_j_idx:torch.tensor, dist_ij:torch.tensor ) \
            -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
            returns triple_dist_ij, triple_dist_ik, triple_dist_jk
            the tensor are ready to be broadcasted with the atom_X_idx tensors
            dist_ij, dist_ik are distance from the center atom i to atoms j and k
            triple_dist_jk is the distance between jk relative to
               the maximum possible (ij+ik) and the minimum possible max(ij,ik)-min(ij,ik)
        """

        n_center = atom_i_idx.shape[0]
        # Expand ij indices to triples with same i but permuting j
        # broadcast_tensors not needed because it is automatic in the distance calc
        triple_j,triple_k = ( atom_j_idx.view(n_center,nNeigh,1),
                              atom_j_idx.view(n_center,1,nNeigh))
        # here we need to broadcast because later torch.cat does not auto-broadcast
        triple_dist_ij, triple_dist_ik = torch.broadcast_tensors(
                                        dist_ij.view(n_center,nNeigh,1),
                                        dist_ij.view(n_center,1,nNeigh))

        triple_dist_jk = self._computeNormalizedJKDistance(data_set, triple_j,triple_k,triple_dist_ij, triple_dist_ik) #jk dist

        return triple_dist_ij, triple_dist_ik, triple_dist_jk

class AngleNet1(nn.Module):
    def __init__(self, conf:Dict[str,Any]):
        super(AngleNet1, self).__init__()

        n_feature = len(list(conf['atom_types'].values())[0])
        fuz = conf['angleNet'].get('fuzz',{'minrad': [], 'stddev': [], 'angleStddev_deg': 0. })
        fuz_min = fuz['minrad'] if isinstance(fuz['minrad'], list) else [fuz['minrad']]
        fuz_std = fuz['stddev'] if isinstance(fuz['stddev'], list) else [fuz['stddev']]
        fuz_angl_std = fuz['angleStddev_deg'] / 180 * math.pi

        aconf = conf['angleNet']
        self.angle_descriptor = aconf.get('angleDescriptor', 'distance')
        self.cutoff = aconf['angularCutoff']

        if self.angle_descriptor == 'distance':
            if fuz_angl_std != 0.:
                raise RuntimeError("angleStddev_deg not supported for distance")
            self.computInput = Compute1AngleInput(n_feature, fuz_min, fuz_std)
        elif self.angle_descriptor == 'angle':
            self.computInput = ComputeRealAngleInput(n_feature, fuz_min, fuz_std, fuz_angl_std)
        elif self.angle_descriptor == 'angle-cutoff':
            self.computInput = ComputeRealAngle2Input(n_feature, self.cutoff, fuz_min, fuz_std, fuz_angl_std)
        else:
            raise RuntimeError(f"unknown angleDescriptor: {aconf['angleDescriptor']}")

        self.mod_dict = nn.ModuleDict()

        nInputs = self.computInput.n_output

        for layer_num, layer_conf in enumerate(aconf['layers']):
            nOut       = layer_conf['nodes']
            bias       = layer_conf['bias']
            activation = layer_conf.get('activation', None)
            batchNorm  = layer_conf.get('batchNorm', None)
            dropOutPct = layer_conf.get('dropOutPct', 0)
            requires_grad = layer_conf.get('requires_grad', True)

            layer = nn.Linear(nInputs, nOut, bias=bias)
            initWeigths(layer, nInputs, nOut, layer_conf.get('initWeight', None))
            if bias: initBias(layer, layer_conf.get('initBias', None))
            name = "AngleNet %d (%d)" % (layer_num, nOut)
            self.mod_dict.add_module(name,layer)

            # activation
            if activation is not None:
                activation = create_activation(
                                activation['type'], activation.get('param',{}))
                name = "AngleNet act%d" % layer_num
                self.mod_dict.add_module(name, activation)

            if  batchNorm is not None:
                name = "AngleNet batchNorm%d" % layer_num
                bNorm = BatchNorm1d(nOut, *batchNorm.get('param',[]))
                self.mod_dict.add_module(name, bNorm)

            if dropOutPct > 0:
                name = "AngleNet dropOut%d" % layer_num
                self.mod_dict.add_module(name, Dropout(dropOutPct / 100., False))

            layer.requires_grad_(requires_grad)
            nInputs = nOut

#         self.add_module("AngleNetDict", self.mod_dict)
        self.n_output:int = nInputs


    def forward(self, ds:DeviceDataSet, inp:Dict[str,torch.tensor]) -> Dict[str,torch.tensor]:
        n_feature = self.computInput.n_output

        if PROFILE > 0:
            # record so trace is easier to follow in chrome://tracing
            #device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            torch.acos_(ds.ZERO)
            torch.erf_(ds.ZERO)

        batch_ang_neighbor_map = inp.pop('batch_ang_neighbor_map')

        desc_list = []
        center_atom_list = []
        for nNeigh, (atom_i_idx, atom_j_idx, dist_ij) in batch_ang_neighbor_map.items():
            center_atom_idx, desc = self.computInput(nNeigh, atom_i_idx, atom_j_idx, dist_ij,
                                         ds.atoms_xyz, ds.atoms_long, ds.atom_types.atom_embedding)
            center_atom_list.append(center_atom_idx)

            in_desc = desc
            nCenter = desc.shape[0]
            nTriple = desc.shape[1]
            desc = desc.reshape(-1,n_feature) # flatten out ntriple dimension for batchnorm

            # we might need to reshape desc so that it has only 2d
            for lyr in self.mod_dict.values():
                desc = lyr(desc)

            desc = desc.reshape(nCenter, nTriple, -1)

            if self.angle_descriptor != 'angle-cutoff':
                # now we need to multiply each with the cutoff function
                # we could try using a bump function here for performance?
                # in_desc[:,:,0:2] are the ij and ik distances
                fc = (0.5 * torch.cos(math.pi * in_desc[:,:,0:2] / self.cutoff) + 0.5)
                fc = fc.prod(dim=-1,keepdim=True)
                desc = desc * fc
            else:
                # dist_ij where already multiplied by cutoff so no more need
                pass

            desc = desc.sum(dim=1)
            #Does the deviation warrant converting to double for the summation?
            #warn(f"ang sum diff: {(desc.sum(dim=1).double() - desc.double().sum(dim=1)).abs().max()*1000}")

            desc_list.append(desc)

        desc            = torch.cat(desc_list)
        center_atom_idx = torch.cat(center_atom_list)
        inp['batch_ang_desc'] = desc
        inp['batch_ang_center_atom_idx'] = center_atom_idx
        return inp


class Compute1AngleInput(nn.Module):
    """
        Computes the Angular input for AngleNet for one set atoms witht he same number of neighbors

           - nCenterAt number of center atoms i, with the same number of neighbors
           - nNeigh * (nNeigh -1): number or triples for this nNeigh
           - 3 (distances ij,ij,jk_norm) + 3 * n_Embedding_features: ijk atoms
        batch_ang_center_map items contain the atom indices of the center atoms i
           of each triple ijk
    """

    def __init__(self, n_feature:int, fuz_min:List[float] = [], fuz_std:List[float] = []):
        super(Compute1AngleInput, self).__init__()

        n_distcolumns = 3
        self.n_output = n_distcolumns + 3 * n_feature
        self.fuz_min = fuz_min
        self.fuz_std = fuz_std


    def forward(self, nNeigh:int, atom_i_idx:torch.tensor, atom_j_idx:torch.tensor, dist_ij:torch.tensor,
                atoms_xyz:torch.tensor, atoms_long:torch.tensor, atom_embedding:nn.Embedding) \
         -> Tuple[torch.tensor, torch.Tensor]:
        fuz_std:List[float] = []
        fuz_min:List[float] = []
        if self.training:
            fuz_std = self.fuz_std
            fuz_min = self.fuz_min

        return Compute1AngleInput.computeInput(nNeigh, atom_i_idx, atom_j_idx, dist_ij,
                                               atoms_xyz, atoms_long, atom_embedding, fuz_min, fuz_std)


#     def _computeNormalizedJKDistance(triple_j:torch.tensor, triple_k:torch.tensor,
#                                      triple_dist_ij:torch.tensor, triple_dist_ik:torch.tensor,
#                                      atoms_xyz:torch.tensor):
    @staticmethod
    @torch.jit.script
    def _computeNormalizedJKDistance(triple_j, triple_k,
                                     triple_dist_ij, triple_dist_ik,
                                     atoms_xyz):
        """
            return jk distance normalizes
        """
        triple_dist_jk = (atoms_xyz[triple_j] - atoms_xyz[triple_k]).norm(p=2, dim=-1)

        # normalize to be between 0 and 1 based on triangle equation:
        # jk distance must be between |dij-dik| and dij+dik
        maxDist = torch.max(triple_dist_ij, triple_dist_ik)
        minDist = torch.min(triple_dist_ij, triple_dist_ik)
        triple_dist_jk = (triple_dist_jk - maxDist + minDist) / (2 * minDist)

        return triple_dist_jk


#     def _computeNormalizedJKDistanceSqr(triple_j:torch.tensor, triple_k:torch.tensor,
#                                         triple_dist_ij:torch.tensor, triple_dist_ik:torch.tensor,
#                                         atoms_xyz:torch.tensor):
    @staticmethod
    @torch.jit.script
    def _computeNormalizedJKDistanceSqr(triple_j, triple_k,
                                        triple_dist_ij, triple_dist_ik,
                                        atoms_xyz):
        """ (return jk distance)^2 normalized """
        triple_dist_sqr_jk = (atoms_xyz[triple_j] - atoms_xyz[triple_k]).pow(2).sum(dim=-1)

        # normalize to be between 0 and 1 based on triangle equation:
        # jk distance must be between (dij-dik)^2 and (dij+dik)^2
        triple_dist_sqr_jk = (triple_dist_sqr_jk - (triple_dist_ij - triple_dist_ik).pow(2)) / (4 * triple_dist_ij * triple_dist_ik)

        return triple_dist_sqr_jk


#     def _computeTriples(nNeigh:int, atom_i_idx:torch.tensor, atom_j_idx:torch.tensor, dist_ij:torch.tensor, atoms_xyz:torch.tensor ) \
#         -> Tuple[torch.tensor, torch.tensor]:
    @staticmethod
    def _computeTriples(nNeigh:int, atom_i_idx, atom_j_idx, dist_ij, atoms_xyz):
        """
            returns triple_dist_ij, triple_dist_ik, triple_dist_jk
            the tensor are ready to be broadcasted with the atom_X_idx tensors
            dist_ij, dist_ik are distance from the center atom i to atoms j and k
            triple_dist_jk is the distance between jk relative to
               the maximum possible (ij+ik) and the minimum possible max(ij,ik)-min(ij,ik)
        """

        n_center = atom_i_idx.shape[0]
        # Expand ij indices to triples with same i but permuting j
        # broadcast_tensors not needed because it is automatic in the distance calc
        triple_j,triple_k = ( atom_j_idx.view(n_center,nNeigh,1),
                              atom_j_idx.view(n_center,1,nNeigh))
        # here we need to broadcast because later torch.cat does not auto-broadcast
        triple_dist_ij, triple_dist_ik = torch.broadcast_tensors(
                                        dist_ij.view(n_center,nNeigh,1),
                                        dist_ij.view(n_center,1,nNeigh))

        triple_dist_jk = Compute1AngleInput._computeNormalizedJKDistance(
                            triple_j,triple_k,triple_dist_ij, triple_dist_ik, atoms_xyz) #jk dist

        return triple_dist_ij, triple_dist_ik, triple_dist_jk


    @staticmethod
    def computeInput(nNeigh:int, atom_i_idx, atom_j_idx, dist_ij, atoms_xyz, atoms_long, atom_embedding,
                    fuz_min:List[float], fuz_std:List[float]):
        """ generate input for network that works on atom triples

            Arguments:
                number of neighbors per center atom
                atom_i_idx cneter atom
                atom_j_idx neighbor atoms within angluar cutoff
                dist_ij
                nNeigh number of neighbor atoms
        """
        device = atoms_long.device

        nCenterAt = atom_i_idx.shape[0] # number of i atoms with nNeigh neighbors
        atom_i_idx= atom_i_idx.to(device, non_blocking=True)
        atom_j_idx= atom_j_idx.to(device, non_blocking=True)
        dist_ij   = dist_ij.to(device, non_blocking=True)

        dist_ij = dist_ij.unsqueeze(-1)
        for fmin, fstd in zip(fuz_min, fuz_std):
            ## apply fuz only to distances larger than fuz_min and normal distributed by the amount dist's are larger than fuz_min
            minDist = torch.clamp(dist_ij - fmin, min=0.)
            dist_ij = dist_ij.addcmul(minDist,torch.randn_like(minDist), value=fstd)
            del minDist

        triple_dist_ij, triple_dist_ik, triple_dist_jk = Compute1AngleInput._computeTriples(nNeigh, atom_i_idx, atom_j_idx, dist_ij, atoms_xyz)
        del dist_ij

        neigh_atom_embed_i = atom_embedding(atoms_long[atom_i_idx,1])
        neigh_atom_embed_j = atom_embedding(atoms_long[atom_j_idx,1])
        neigh_atom_embed_i, neigh_atom_embed_j, neigh_atom_embed_k = torch.broadcast_tensors(
                neigh_atom_embed_i.view(nCenterAt,1,1,-1),
                neigh_atom_embed_j.view(nCenterAt,nNeigh,1,-1),
                neigh_atom_embed_j.view(nCenterAt,1,nNeigh,-1))
        triple_dist_ij = triple_dist_ij.unsqueeze(-1)
        triple_dist_ik = triple_dist_ik.unsqueeze(-1)
        triple_dist_jk = triple_dist_jk.unsqueeze(-1)
        ang_desc = torch.cat((
            triple_dist_ij, triple_dist_ik, triple_dist_jk,
            neigh_atom_embed_i,
            neigh_atom_embed_j/triple_dist_ij, neigh_atom_embed_k/triple_dist_ik), dim=-1)

        # filter out diagonal elements in jk
        filtr = (1-torch.eye(nNeigh,dtype=torch.uint8, device=device)).bool()
        # below gets lower loss because it includes diagonal elements????
        #filtr = torch.ones((nNeigh,nNeigh),dtype=torch.uint8, device=ang_desc.device).bool()
        ang_desc = ang_desc[:,filtr,:]

        return atom_i_idx.view(-1), ang_desc

class ComputeRealAngleInput(nn.Module):
    """
        Computes the Angular input for AngleNet for one set atoms with the same number of neighbors

           - nCenterAt number of center atoms i, with the same number of neighbors
           - nNeigh * (nNeigh -1): number or triples for this nNeigh
           - 2 (distances ij,ij and one angle jk) + 3 * n_Embedding_features: ijk atoms
        batch_ang_center_map items contain the atom indices of the center atoms i
           of each triple ijk
    """

    def __init__(self, n_feature:int, fuz_min:List[float] = [], fuz_std:List[float] = [], fuz_angl_std = 0.):
        super(ComputeRealAngleInput, self).__init__()

        n_distcolumns = 3
        self.n_output = n_distcolumns + 3 * n_feature
        self.fuz_min = fuz_min
        self.fuz_std = fuz_std
        self.fuz_angl_std = fuz_angl_std

    def forward(self, nNeigh:int, atom_i_idx:torch.tensor, atom_j_idx:torch.tensor, dist_ij:torch.tensor,
                atoms_xyz:torch.tensor, atoms_long:torch.tensor, atom_embedding:nn.Embedding) \
         -> Tuple[torch.tensor, torch.Tensor]:

        return self.computeInput(nNeigh, atom_i_idx, atom_j_idx, dist_ij,
                                 atoms_xyz, atoms_long, atom_embedding)


    #@torch.jit.script
    def _computeAngleIJK(self, atom_i_idx, triple_j, triple_k, atoms_xyz):
        """
            return jk angles
        """
        vec_ij = atoms_xyz[triple_j] - atoms_xyz[atom_i_idx.unsqueeze(1)]
        vec_ik = atoms_xyz[triple_k] - atoms_xyz[atom_i_idx.unsqueeze(-1)]
        ang = torch.acos(torch.nn.functional.cosine_similarity(vec_ij,vec_ik, dim=-1)*0.9999)

        if self.fuz_angl_std != 0. and self.training:
            ang = ang.addcmul(ang, torch.randn_like(ang), value=self.fuz_angl_std).abs()
            ang[ang > math.pi] = 2 * math.pi - ang[ ang > math.pi ]
        return ang

    # @torch.jit.script
    def _computeTriples(self, nNeigh:int, atom_i_idx, atom_j_idx, dist_ij, atoms_xyz):
        """
            returns triple_dist_ij, triple_dist_ik, triple_ang_jk
            the tensor are ready to be broadcasted with the atom_X_idx tensors
            dist_ij, dist_ik are distance from the center atom i to atoms j and k
            triple_ang_jk is the angle between i and j,k
        """

        n_center = atom_i_idx.shape[0]
        # Expand ij indices to triples with same i but permuting j
        # broadcast_tensors not needed because it is automatic in the distance calc
        triple_j,triple_k = ( atom_j_idx.view(n_center,nNeigh,1),
                              atom_j_idx.view(n_center,1,nNeigh))
        # here we need to broadcast because later torch.cat does not auto-broadcast
        triple_dist_ij, triple_dist_ik = torch.broadcast_tensors(
                                        dist_ij.view(n_center,nNeigh,1),
                                        dist_ij.view(n_center,1,nNeigh))

        triple_angle_jk = self._computeAngleIJK(atom_i_idx,
                            triple_j,triple_k, atoms_xyz) #jk angle

        return triple_dist_ij, triple_dist_ik, triple_angle_jk


    #@torch.jit.script
    def computeInput(self, nNeigh:int, atom_i_idx, atom_j_idx, dist_ij, atoms_xyz, atoms_long, atom_embedding):
        """ generate input for network that works on atom triples
            Arguments:
                number of neighbors per center atom
                atom_i_idx cneter atom
                atom_j_idx neighbor atoms within angluar cutoff
                dist_ij distance matrix
        """
        device = atoms_long.device

        dist_ij   = dist_ij.to(device, non_blocking=True)
        nCenterAt = atom_i_idx.shape[0] # number of i atoms with nNeigh neighbors
        atom_i_idx= atom_i_idx.to(device, non_blocking=True)
        atom_j_idx= atom_j_idx.to(device, non_blocking=True)

        if self.training:
            for fmin, fstd in zip(self.fuz_min, self.fuz_std):
                ## apply fuz only to distances larger than fuz_min and normal distributed by the amount dist's are larger than fuz_min
                minDist = torch.clamp(dist_ij - fmin, min=0.)
                dist_ij = dist_ij.addcmul(minDist,torch.randn_like(minDist), value=fstd)
                del minDist

        triple_dist_ij, triple_dist_ik, triple_angle_jk = self._computeTriples(nNeigh, atom_i_idx, atom_j_idx, dist_ij, atoms_xyz)
        del dist_ij

        neigh_atom_embed_i = atom_embedding(atoms_long[atom_i_idx,1])
        neigh_atom_embed_j = atom_embedding(atoms_long[atom_j_idx,1])
        neigh_atom_embed_i, neigh_atom_embed_j, neigh_atom_embed_k = torch.broadcast_tensors(
                neigh_atom_embed_i.view(nCenterAt,1,1,-1),
                neigh_atom_embed_j.view(nCenterAt,nNeigh,1,-1),
                neigh_atom_embed_j.view(nCenterAt,1,nNeigh,-1))
        triple_dist_ij = triple_dist_ij.unsqueeze(-1)
        triple_dist_ik = triple_dist_ik.unsqueeze(-1)
        triple_angle_jk = triple_angle_jk.unsqueeze(-1)
        ang_desc = torch.cat((
            triple_dist_ij, triple_dist_ik, triple_angle_jk,
            neigh_atom_embed_i,
            neigh_atom_embed_j/triple_dist_ij, neigh_atom_embed_k/triple_dist_ik), dim=-1)

        # filter out diagonal elements in jk
        filtr = (1-torch.eye(nNeigh,dtype=torch.uint8, device=device)).bool()
        # below gets lower loss because it includes diagonal elements????
        #filtr = torch.ones((nNeigh,nNeigh),dtype=torch.uint8, device=ang_desc.device).bool()
        ang_desc = ang_desc[:,filtr,:]

        return atom_i_idx.view(-1), ang_desc


class ComputeRealAngle2Input(ComputeRealAngleInput):
    """
        Computes the Angular input for AngleNet for one set atoms with the same number of neighbors
        This implementation multiplies the distance with the cutoff function in order to speedup network

           - nCenterAt number of center atoms i, with the same number of neighbors
           - nNeigh * (nNeigh -1): number or triples for this nNeigh
           - 2 (distances ij,ij and one angle jk) + 3 * n_Embedding_features: ijk atoms
        batch_ang_center_map items contain the atom indices of the center atoms i
           of each triple ijk
    """

    def __init__(self, n_feature:int, cutoff:float, fuz_min:List[float] = [], fuz_std:List[float] = [], fuz_angl_std = 0.):
        super().__init__(n_feature, fuz_min, fuz_std, fuz_angl_std)

        # one embedding per atom plus angle
        self.n_output = 3 * n_feature + 1

        self.cutoff =  cutoff


    def _computeTriples(self, nNeigh:int, atom_i_idx, atom_j_idx, dist_ij, atoms_xyz):
        """
            returns triple_dist_ij, triple_dist_ik, triple_ang_jk
            the tensor are ready to be broadcasted with the atom_X_idx tensors
            dist_ij, dist_ik are 1/distance from the center atom i to atoms j and k
                times the cutoff function that ensures that for dist > cutoff dist and fist` are 0
            triple_ang_jk is the angle between i and j,k
        """

        n_center = atom_i_idx.shape[0]
        # Expand ij indices to triples with same i but permuting j
        # broadcast_tensors not needed because it is automatic in the distance calc
        triple_j,triple_k = ( atom_j_idx.view(n_center,nNeigh,1),
                              atom_j_idx.view(n_center,1,nNeigh))

        # multiply dist with cutoff function that ensures derivatives are 0 at cutoff
        fc = (0.5 * torch.cos(math.pi * dist_ij / self.cutoff) + 0.5)
        dist_ij = 1/dist_ij * fc

        # here we need to broadcast because later torch.cat does not auto-broadcast
        triple_dist_ij, triple_dist_ik = torch.broadcast_tensors(
                                        dist_ij.view(n_center,nNeigh,1),
                                        dist_ij.view(n_center,1,nNeigh))

        triple_angle_jk = self._computeAngleIJK(atom_i_idx,
                            triple_j,triple_k, atoms_xyz) #jk angle

        return triple_dist_ij, triple_dist_ik, triple_angle_jk


    # @torch.jit.script
    def computeInput(self, nNeigh:int, atom_i_idx, atom_j_idx, dist_ij, atoms_xyz, atoms_long, atom_embedding):
        """ generate input for network that works on atom triples

            Arguments:
                number of neighbors per center atom
                atom_i_idx center atom
                atom_j_idx neighbor atoms within angular cutoff
                dist_ij distance matrix
                nNeigh: number of neighbor atoms
        """
        device = atoms_long.device

        dist_ij   = dist_ij.to(device, non_blocking=True)
        nCenterAt = atom_i_idx.shape[0] # number of i atoms with nNeigh neighbors
        atom_i_idx= atom_i_idx.to(device, non_blocking=True)
        atom_j_idx= atom_j_idx.to(device, non_blocking=True)

        if self.training:
            for fmin, fstd in zip(self.fuz_min, self.fuz_std):
                ## apply fuz only to distances larger than fuz_min and normal distributed by the amount dist's are larger than fuz_min
                minDist = torch.clamp(dist_ij - fmin, min=0.)
                dist_ij = dist_ij.addcmul(minDist,torch.randn_like(minDist), value=fstd)
                del minDist

        triple_dist_ij, triple_dist_ik, triple_angle_jk \
            = self._computeTriples(nNeigh, atom_i_idx, atom_j_idx, dist_ij, atoms_xyz)
        del dist_ij

        neigh_atom_embed_i = atom_embedding(atoms_long[atom_i_idx,1])
        neigh_atom_embed_j = atom_embedding(atoms_long[atom_j_idx,1])
        neigh_atom_embed_i, neigh_atom_embed_j, neigh_atom_embed_k = torch.broadcast_tensors(
                neigh_atom_embed_i.view(nCenterAt,1,1,-1),
                neigh_atom_embed_j.view(nCenterAt,nNeigh,1,-1),
                neigh_atom_embed_j.view(nCenterAt,1,nNeigh,-1))
        triple_dist_ij = triple_dist_ij.unsqueeze(-1)
        triple_dist_ik = triple_dist_ik.unsqueeze(-1)
        triple_angle_jk = triple_angle_jk.unsqueeze(-1)
        ang_desc = torch.cat((
            triple_angle_jk,
            neigh_atom_embed_i,
            neigh_atom_embed_j*triple_dist_ij, neigh_atom_embed_k*triple_dist_ik), dim=-1)

        # filter out diagonal elements in jk
        filtr = (1-torch.eye(nNeigh,dtype=torch.uint8, device=device)).bool()
        # below gets lower loss because it includes diagonal elements????
        #filtr = torch.ones((nNeigh,nNeigh),dtype=torch.uint8, device=ang_desc.device).bool()
        ang_desc = ang_desc[:,filtr,:]

        return atom_i_idx.view(-1), ang_desc
