from ase.units import Hartree, kcal
from cdd_chem.util import constants
from ml_qm.ANIMol import ANIMol
from ml_qm import AtomInfo
from ml_qm.pt import torch_util as tu
import ANI.lib.pyanitools as pya
from ase.units import mol as mol_unit
import cdd_chem.util.debug.memory
from typing import Dict, Sequence, Tuple, Optional

import torch.cuda
import gc
import glob
import numpy as np
import copy
import os
import pickle
import gzip
import h5py
import logging

from os import path
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611
from cdd_chem.util.debug.memory import print_memory_usage
from t_opt.atom_info import NameToNum

log = logging.getLogger(__name__)



class AtomTypes:
    """ Embedding of atoms """

    atom_embedding:torch.nn.Embedding
    atom_vdw:torch.tensor
    n_atom_type:int
    n_feature:int
    at_prop_names_2_col:Dict[str,int]

    def __init__(self, atom_types, atom_embedding_conf):
        self.n_atom_type = len(atom_types)
        self.n_feature = len(list(atom_types.values())[0])
        self.atom_vdw = torch.tensor(AtomInfo.NumToVDWRadius)
        max_atom_num = max([AtomInfo.NameToNum[k] for k in atom_types.keys()])

        at_props = torch.full((max_atom_num+1, self.n_feature),-1, dtype=torch.float32)
        at_prop_names = list(list(atom_types.values())[0].keys())
        self.at_prop_names_2_col = { nam:i for (i,nam) in enumerate(at_prop_names)}

        for at_sym, at_prop in atom_types.items():
            at_num = AtomInfo.NameToNum[at_sym]
            at_props[at_num] = torch.tensor(list(at_prop.values()), dtype=torch.float32)

        # normalize by avg and stdev if given
        if atom_embedding_conf and atom_embedding_conf.get('normalization', None):
            for nam, stat in atom_embedding_conf['normalization'].items():
                if nam in self.at_prop_names_2_col:
                    prop_col = self.at_prop_names_2_col[nam]
                    at_props[:, prop_col] = (
                        at_props[:, prop_col] - stat['avg']) / stat['sigma']

        self.atom_embedding = torch.nn.Embedding.from_pretrained(at_props, freeze=True)

    def to_(self, device):
        """ move to device """
        self.atom_embedding = self.atom_embedding.to(device)
        self.atom_vdw       = self.atom_vdw.to(device)


    def to(self, device):
        """ clone to device """
        newAt = copy.deepcopy(self)  # deepcopy needed because to() does not create new object
        newAt.atom_embedding = newAt.atom_embedding.to(device)
        newAt.atom_vdw       = self.atom_vdw.to(device)

        return newAt


class DataSet:
    """
        A DataSet contains these Tensors:
        conformations:  size: nconf,  2   float32 containing [Energy, n_atom]
        atoms_long:     size: nAtoms, 2   long    containing [conf_idx, atom_num]
        atoms_xyz:      size: nAtoms, 3   float32 containing [x,y,z]

        rad_dist_map['atom_ij_idx'] tensor with 2 columns: atoms_ij
        rad_dist_map['dist_ij']: distance_ij

        ang_neighbor_map:
        containing TensorBuffers Each TB is keyed by the number of neighbors per atom
        The angular cutoff was applied when computing the neighbors
        i_buffer, j_buffer, d_buffer = ds.ang_neighbor_map[nNeigh]
            i_buffer.shape = number_of_Center_atoms,1 contains the atom idx for atom i
            j_buffer.shape = number_of_Center_atoms,nNeigh contains the atom idx for each neighbor j
            d_buffer.shape = number_of_Center_atoms,nNeigh contains the distance ij
                ij = torch.stack(torch.broadcast_tensors(i_buffer.buffer,j_buffer.buffer), dim=2)
             returns a list of atom indices ij for each neighbor ij
             ij.shape = number_of_Center_atoms, nNeigh, 2

        and
        atom_types atom_type information as store in AtomTypes
        n_confs:int
        conf:Dict[str,object] configuration file

        Note: some ints are stored as float to allow for faster slizing
    """

    def __init__(self, conf:Dict, device = None):
        """ if cuda is available you can provide the device and it will be used to
            Speed up the dataset creation
        """

        self.atom_types = AtomTypes(conf['atom_types'], conf['atom_embedding'])
        self.conf = conf
        self.device = device

        self.conformations = tu.TensorBuffer((conf['confChunkSize'],2), dtype=torch.float32)
        self.atoms_long    = tu.TensorBuffer((conf['atomChunkSize'],2), dtype=torch.long)
        self.atoms_xyz     = tu.TensorBuffer((conf['atomChunkSize'],3), dtype=torch.float32)

            # (atom_idx, j_atom_idx, dist_ij), cutoff=angRadCutoff; indexes and distances keyed by n_neighbors
        self.ang_neighbor_map: Dict[int,Tuple[torch.tensor,torch.tensor,torch.tensor]] = {}

        self.rad_dist_map: Dict[str,torch.tensor] = {} # ((atom_idx, j_atom_idx), dist_ij)  cutoff=radCutoff
        self.energy_corrections = DataSet._get_energy_corrections(conf, conf['atom_types'])

        tconf                   = self.conf['trainData']
        self.skip_factor        = tconf.get('skipFactor',1)
        self.atomizationE_range = tconf['atomizationERange']

        self.n_confs = 0
        self.n_atoms = 0
        self.rad_dist_map['atom_ij_idx'] = tu.TensorBuffer((self.conf['pairChunkSize'],2), dtype=torch.long)
        self.rad_dist_map['dist_ij']     = tu.TensorBuffer((self.conf['pairChunkSize']), dtype=torch.float16)


    def load_conf_data(self):
        # overwrite if there are tconf specific settings
        tconf = self.conf['trainData']
        pickle_file = tconf.get('pickleFile',None)
        if pickle_file:
            pickle_file = constants.replace_consts_and_env(pickle_file)
            if os.path.exists(pickle_file):
                log.info(f"loading data set from: {pickle_file}")
                pickl_dict = self.read_pickle(pickle_file)

                self.conformations = pickl_dict['conformations']
                self.atoms_long    = pickl_dict['atoms_long']
                self.atoms_xyz     = pickl_dict['atoms_xyz']
                self.ang_neighbor_map = pickl_dict['ang_neighbor_map'] # (atom_idx, j_atom_idx, dist_ij) indexes and distances keyed by n_neighbors
                self.rad_dist_map     = pickl_dict['rad_dist_map'] # atom_ij_idx, dist_ij) with radialNet cutoff
                self.n_confs = pickl_dict['n_confs']
                self.n_atoms = pickl_dict['n_atoms']

                log.warning("Sizes read from pickle:")
                log.warning(f"  n_confs: {self.n_confs}")
                log.warning(f"  n_atoms: {self.n_atoms}")
                log.warning(f"  conformations: {self.conformations.buffer.shape}")
                log.warning(f"  atoms_long: {self.atoms_long.buffer.shape}")
                log.warning(f"  atoms_xyz: {self.atoms_xyz.buffer.shape}")
                print_memory_usage()
                if log.isEnabledFor(logging.DEBUG):
                    for k,(i,j,d) in self.ang_neighbor_map.items():
                        log.debug(f"  neigh[{k}]: {i.buffer.shape},{j.buffer.shape}, {d.buffer.shape}")
                        assert k == j.buffer.shape[1] and k == d.buffer.shape[1]
                        assert i.buffer.shape[0] == j.buffer.shape[0] and i.buffer.shape[0] == d.buffer.shape[0]
                    aij = self.rad_dist_map['atom_ij_idx'].buffer
                    dij = self.rad_dist_map['dist_ij'].buffer
                    log.debug(f"  rad_dist_map: {aij.shape} {dij.shape}")
                return

        self.skip_factor        = tconf.get('skipFactor',1)
        self.atomizationE_range = tconf['atomizationERange']

        data_dir = constants.replace_consts_and_env(tconf['dataDir'])
        in_pattern = "%s/%s" % (data_dir, tconf['iFile'])

        if tconf['type'] == 'ANI-2_201910':
            self.load_ani201910(in_pattern)
        if tconf['type'] == 'ANI-2_202001':
            self.load_ani202001(in_pattern)
        elif tconf['type'] == 'ANI-2':
            self.load_ani2(data_dir, in_pattern)
        else:
            raise RuntimeError("Unknown train data type: {tconf['type']}")


    def read_pickle(self, pickle_file):
        with gzip.open(pickle_file, 'rb') as infile:
            pickl_dict = pickle.load(infile)

            if 'conformations' in pickl_dict:
                return pickl_dict   # this was written in old memory hungry format


            pickl_dict['conformations'] = pickle.load(infile)
            pickl_dict['atoms_long']    = pickle.load(infile)
            pickl_dict['atoms_xyz']     = pickle.load(infile)

            ang_neighbor_map = {}
            kv = pickle.load(infile)
            while kv:
                (k,v) = kv
                ang_neighbor_map[k] = v
                kv = pickle.load(infile)
            pickl_dict['ang_neighbor_map'] = ang_neighbor_map

            rad_dist_map = {}
            kv = pickle.load(infile)
            while kv:
                (k,v) = kv
                rad_dist_map[k] = v
                kv = pickle.load(infile)
            pickl_dict['rad_dist_map'] = rad_dist_map

        return pickl_dict


    def load_ani2(self, data_dir:str, in_pattern:str ):
        files = glob.glob(in_pattern)
        files.sort()
        cnt_conf = 0

        for f in files:
            log.info("Processing: %s" % f)

            adl = pya.anidataloader(f)
            f = path.split(f)[1]
            for rec in adl:

                # Extract the data
                atom_types= list(AtomInfo.NameToNum[a] for a in rec['species'])
                confs_mol = rec['coordinates']
                e         = rec['energies']
                e         = e * Hartree*mol_unit/kcal

                self.add_conformers( atom_types, confs_mol, e )

                nconf = e.shape[0]
                cnt_conf += nconf
                if cnt_conf // 100000 != (cnt_conf - nconf) // 100000:
                    warn(f"conf {cnt_conf}")


    def load_ani201910(self, in_pattern:str ):
        self._load_ani2_new("atomic_numbers", None, "wb97x_dz.energy", "coordinates", "wb97x_dz.forces",
                           in_pattern)


    def load_ani202001(self, in_pattern:str ):
        self._load_ani2_new(None, "species", "energies", "coordinates", "forces",
                           in_pattern)


    def _load_ani2_new(self, at_num_field:Optional[str], at_sym_field:Optional[str],
                       energy_field:str, coord_field:str, force_field:str,
                       in_pattern:str ):

        files = glob.glob(in_pattern)
        files.sort()

        if len(files) == 0:
            log.warning(f"No files match pattern: {in_pattern}")

        cnt_conf = 0
        for f in files:
            log.info("Processing: %s" % f)
            inFile = h5py.File(f, "r")

            for key,item in inFile.items():
                name = str(key) # noqa: F841

                if at_num_field:
                    atom_types = item[at_num_field][()] # np.array(uint8)
                else:
                    atom_types = [ NameToNum[s.decode('ascii')] for s in item[at_sym_field][()] ] # np.array(uint8)

                e = item[energy_field][()] # np.array((nMol,nAt])
                confs_mol = item[coord_field][()] # np.array((nMol,nAt,3])
                #at_force = item[force_field][()] # np.array((nMol,nAt,3])

                e *= Hartree*mol_unit/kcal

                self.add_conformers( atom_types, confs_mol, e )

                nconf = e.shape[0]
                cnt_conf += nconf
                if cnt_conf // 100000 != (cnt_conf - nconf) // 100000:
                    warn(f"conf {cnt_conf}")


    def finalize(self):
        """ Only the conf and atom tensors will be put on device as the
            neighbor info will not fit
        """
        assert self.conformations.nRows > 0
        self.conformations.finalize("conformations")
        self.atoms_long.finalize("atoms_long")
        self.atoms_xyz.finalize("atoms_xyz")

        self.rad_dist_map['atom_ij_idx'].finalize()
        self.rad_dist_map['dist_ij'].finalize()

        for nNeigh, (itb, jtb, dtb) in self.ang_neighbor_map.items():
            itb.finalize(f'neigh_i[{nNeigh}]')
            jtb.finalize() # do not name as same size as i TB f'neigh_j[{nNeigh}]')
            dtb.finalize() # do not name as same size as i TB f'dist_ij[{nNeigh}]')

        tconf = self.conf['trainData']
        pickle_file = tconf.get('pickleFile',None)
        if pickle_file:
            pickle_file = constants.replace_consts_and_env(pickle_file)
            if not os.path.exists(pickle_file):
                self.write_Pickle(pickle_file)

                if log.isEnabledFor(logging.DEBUG):
                    log.debug("Sizes written to pickle:")
                    log.debug(f"  n_confs: {self.n_confs}")
                    log.debug(f"  n_atoms: {self.n_atoms}")
                    log.debug(f"  conformations: {self.conformations.buffer.shape}")
                    log.debug(f"  atoms_long: {self.atoms_long.buffer.shape}")
                    log.debug(f"  atoms_xyz: {self.atoms_xyz.buffer.shape}")
                    for k,(i,j,d) in self.ang_neighbor_map.items():
                        log.debug(f"  neigh[{k}]: {i.buffer.shape},{j.buffer.shape}, {d.buffer.shape}")
                        assert k == j.buffer.shape[1] and k == d.buffer.shape[1]
                        assert i.buffer.shape[0] == j.buffer.shape[0] and i.buffer.shape[0] == d.buffer.shape[0]
                    aij = self.rad_dist_map['atom_ij_idx'].buffer
                    dij = self.rad_dist_map['dist_ij'].buffer
                    log.debug(f"  rad_dist_map: {aij.shape} {dij.shape}")


    def write_Pickle(self, pickle_file):
        with gzip.open(pickle_file, mode='wb') as out:
            pickle.dump({'n_confs':self.n_confs,
                         'n_atoms':self.n_atoms}, out, protocol=4)
            pickle.dump(self.conformations, out, protocol=4)
            pickle.dump(self.atoms_long,    out, protocol=4)
            pickle.dump(self.atoms_xyz,     out, protocol=4)

            for k,v in self.ang_neighbor_map.items():
                pickle.dump((k,v), out, protocol=4)
            pickle.dump(None, out, protocol=4)

            for k,v in  self.rad_dist_map.items():
                pickle.dump((k,v), out, protocol=4)
            pickle.dump(None, out, protocol=4)


    def write_old_Pickle(self, pickle_file):
        pick_dict = {'n_confs':self.n_confs,
            'n_atoms':self.n_atoms,
            'conformations':self.conformations,
            'atoms_long':self.atoms_long,
            'atoms_xyz':self.atoms_xyz,
            'ang_neighbor_map':self.ang_neighbor_map,
            'rad_dist_map':self.rad_dist_map}
        with gzip.open(pickle_file, mode='wb') as out:
            pickle.dump(pick_dict, out, protocol=4)


    def add_conformers(self, atom_types:Sequence[int], confs_mol:np.ndarray, e:np.ndarray):
        assert confs_mol.shape[0] == e.shape[0]
        assert confs_mol.shape[1] == len(atom_types)

        confs_mol = confs_mol[::self.skip_factor]
        e         = e[::self.skip_factor]
        # normalize atomization energies by atomic atomization energies
        e_norm = DataSet.e_normalization_diff(e, self.energy_corrections, atom_types)

        if self.atomizationE_range:
            min_E, max_E = self.atomizationE_range
            is_good=(e_norm >= min_E) & (e_norm <= max_E)
            e_norm = e_norm[is_good]
            confs_mol = confs_mol[is_good]
        if len(e_norm) == 0: return

        n_conf     = e_norm.shape[0]

        atom_types_tnsr = torch.tensor(atom_types, dtype=torch.long, device=self.device)
        n_atoms_per_mol = len(atom_types)

        e_norm      = torch.tensor(e_norm,dtype=torch.float32, device=self.device)
        confs_molpt = torch.tensor(confs_mol,dtype=torch.float32, device=self.device)
        del confs_mol

        conf_idx  = torch.arange(self.n_confs, self.n_confs + n_conf, dtype=torch.long, device=self.device)
        atom_idx  = torch.arange(self.n_atoms, self.n_atoms + n_conf * n_atoms_per_mol, dtype=torch.int32, device=self.device)
        atom_idx  = atom_idx.reshape(n_conf,n_atoms_per_mol)

        ang_Cutoff = self.conf['angleNet']['angularCutoff']
        self._compute_angular_neighbors(n_atoms_per_mol, confs_molpt, atom_idx, ang_Cutoff)

        rad_Cutoff = self.conf['radialNet']['radialCutoff']
        self._compute_rad_dist(n_atoms_per_mol, confs_molpt, atom_idx, rad_Cutoff)

        self.atoms_long.append(torch.stack((
                conf_idx.repeat_interleave(n_atoms_per_mol),
                atom_types_tnsr.repeat(n_conf)),
                dim=1).cpu())
        self.atoms_xyz.append( confs_molpt.reshape(-1,3).cpu() )

        self.conformations.append(torch.stack((
                e_norm,
                torch.full((n_conf,),n_atoms_per_mol, dtype=torch.float32, device=self.device)),
                dim=1).cpu())

        self.n_confs += n_conf
        self.n_atoms += n_conf * n_atoms_per_mol

    @staticmethod
    def e_normalization_diff(e, energyCorrections, atomTypes):
        eCorrection = sum( [ ANIMol.atomEnergies[atNum] for atNum in atomTypes ])
        if energyCorrections is not None:
            for atNum in atomTypes:
                eCorrection += energyCorrections[atNum]

        return e - eCorrection

    @staticmethod
    def _get_energy_corrections(conf, atom_names):
        energyCorrections = conf.get('energyCorrections', None)
        if energyCorrections is None: return None

        atomNums = [AtomInfo.NameToNum[at] for at in atom_names]
        atomNum2ECorrection = [0.] * (max(atomNums) + 1)

        if energyCorrections is not None:
            for i, at in enumerate(atomNums):
                atomNum2ECorrection[at] = energyCorrections[i]

        return atomNum2ECorrection

    def _compute_rad_dist(self,
                          n_atom_per_mol:int, confs_xyz:torch.tensor, atom_idx:torch.tensor,
                          rad_Cutoff:float):
        dist = (confs_xyz.unsqueeze(-2) - confs_xyz.unsqueeze(-3)).norm(dim=-1)
        dist = dist.view(-1)
        i_atom_idx     = atom_idx.repeat_interleave(n_atom_per_mol,dim=-1).view(-1)
        j_atom_idx     = atom_idx.repeat_interleave(n_atom_per_mol,dim=0).view(-1)

        is_good = (dist < rad_Cutoff) & (dist > 0.1)
        dist = dist[is_good].to(dtype=torch.float16)
        i_atom_idx = i_atom_idx[is_good]
        j_atom_idx = j_atom_idx[is_good]

        self.rad_dist_map['atom_ij_idx'].append(torch.stack((i_atom_idx,j_atom_idx),dim=1).cpu())
        self.rad_dist_map['dist_ij'].append(dist.cpu())


    def _compute_angular_neighbors(self,
                          n_atom_per_mol:int, confs_xyz:torch.tensor,
                          atom_idx:torch.tensor, rad_Cutoff:float):
        """
           Compute all pairwise similarities within each conformation, than
           remove those with dist > rad_Cutoff and return with indices

           @param
               ang_neighbor_map: a map of TensorBuffer's keyed by number of neighbors around each atom
               to collect a list:
                    i_idx_buffer, j_idx_buffer, dist_buffer
               n_atom_per_mol: atoms per molecules
               confs_xyz: tensor (n_Conf, 3)   xyz coordinates
               atom_idx:  tensor n_conf, n_atom_per_mol
               rad_Cutoff: radial cutoff

        """
        dist = (confs_xyz.unsqueeze(-2) - confs_xyz.unsqueeze(-3)).norm(dim=-1)
        dist = dist.view(-1)
        i_atom_idx     = atom_idx.repeat_interleave(n_atom_per_mol,dim=-1).view(-1)
        j_atom_idx     = atom_idx.repeat_interleave(n_atom_per_mol,dim=0).view(-1)

        is_good = (dist < rad_Cutoff) & (dist > 0.1)

        dist = dist[is_good].to(dtype=torch.float16)
        i_atom_idx = i_atom_idx[is_good]
        j_atom_idx = j_atom_idx[is_good]

        # split by atom so we can create map by neighbor count
        # i_atom_idx is already sorted but not sure if unique maintains that
        i_atom_idx,counts = i_atom_idx.unique(sorted=True,return_counts=True)

        counts = counts.tolist()
        i_atom_idx = i_atom_idx.split(1)
        j_atom_idx = j_atom_idx.split(counts)
        dist = dist.split(counts)
        ang_neighbor_map = self.ang_neighbor_map

        for i,j,d in zip(i_atom_idx, j_atom_idx, dist):
            nNeigh = d.shape[0]
            if nNeigh < 2: continue

            if nNeigh not in ang_neighbor_map:
                i_idx_buffer = tu.TensorBuffer((self.conf['pairChunkSize'],1),      dtype=torch.int32)
                j_idx_buffer = tu.TensorBuffer((self.conf['pairChunkSize'],nNeigh), dtype=torch.int32)
                dist_buffer  = tu.TensorBuffer((self.conf['pairChunkSize'],nNeigh), dtype=torch.float16)
                ang_neighbor_map[nNeigh] = (i_idx_buffer,j_idx_buffer,dist_buffer)

            i_idx_buffer,j_idx_buffer,dist_buffer = ang_neighbor_map[nNeigh]

            #warn(nNeigh)
            i_idx_buffer.append(i.view(1,-1).cpu())
            j_idx_buffer.append(j.view(1,-1).cpu())
            dist_buffer.append(d.view(1,-1).cpu())

        return


    def to(self,device):
        """ Create a DeviceDataSet with the data that is permanently loaded on the GPU:
        """
        if log.isEnabledFor(logging.DEBUG):
            if "cuda" in str(device):
                torch.cuda.empty_cache()
                log.debug(f"Before GPU loading: mem allocated: {torch.cuda.memory_allocated(device)} "
                          f"mem cached: {torch.cuda.memory_reserved(device)} "
                          f"max allocated: {torch.cuda.max_memory_allocated(device)} "
                          f"max mem cached:{torch.cuda.max_memory_reserved(device)}")
            log.debug(f"CPU RSS(gb): {cdd_chem.util.debug.memory.get_rss_gb()}")


        dds = DeviceDataSet(self.atom_types, self.conformations.buffer, self.atoms_long.buffer, self.atoms_xyz.buffer)
        del self.atoms_xyz
        dds.to_(device)
        return dds


class DeviceDataSet:
    """ container of device located items that are needed for DistNet
    """


    def __init__(self, atom_types:AtomTypes, conf_info:torch.tensor, atom_info:torch.tensor, xyz:torch.tensor):
        """
        :param atom_types:
        :param conf_info:  tensor[NConf,2]  first column is energy, second is number of atoms
        :param atom_info:  tensor[nAt,2]    fist column is conf_idx, second column is atom type
        :param xyz:        tensor[nAt,3]    xyz coordinates
        """
        self.n_confs = conf_info.shape[0]
        self.n_atoms = atom_info.shape[0]
        self.atom_types = atom_types

        self.conformations = conf_info
        self.atoms_long    = atom_info
        self.atoms_xyz     = xyz
        self.device = self.atoms_xyz.device

        self.ZERO    = torch.tensor([0.])
        self.ONE     = torch.tensor([1.])
        self.NEGONE  = torch.tensor([-1.])
        self.HALF    = torch.tensor([0.5])

    def to_(self,device):
        """ move to device """
        self.atom_types.to_(device)

        self.conformations = self.conformations.to(device)
        self.atoms_long    = self.atoms_long.to(device)
        self.atoms_xyz     = self.atoms_xyz.to(device)

        self.ZERO    = torch.tensor([0.], device=device)
        self.ONE     = torch.tensor([1.],  device=device)
        self.NEGONE  = torch.tensor([-1.], device=device)
        self.HALF    = torch.tensor([0.5], device=device)

        self.device = device
        gc.collect()


    def to(self,device):
        """ clone to device """
        newDS = copy.copy(self)
        newDS.atom_types = newDS.atom_types.to(device)

        newDS.conformations = newDS.conformations.to(device)
        newDS.atoms_long    = newDS.atoms_long.to(device)
        newDS.atoms_xyz     = newDS.atoms_xyz.to(device)

        newDS.ZERO    = torch.tensor([0.],  device=device)
        newDS.ONE     = torch.tensor([1.],  device=device)
        newDS.NEGONE  = torch.tensor([-1.], device=device)
        newDS.HALF    = torch.tensor([0.5], device=device)

        newDS.device=device

        return newDS
