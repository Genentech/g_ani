"""
Created on Jul 30, 2019

@author: albertgo
"""
import torch
from typing import List, Collection, Dict
import queue
import threading
import logging
from datetime import datetime
from threading import Barrier
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611
from ml_qm.distNet.data_set import DataSet
log = logging.getLogger(__name__)


class DataLoader:
    """
       Load ANI data for DistNet
    """

    def __init__(self, data_set:DataSet, conf_ids:torch.Tensor,
                 batch_size:int, set_type='train', drop_last:bool = False,
                 nCPU:int = 2, device=None):
        """
           Argument:
               set_type: if == 'train' than random sampling will be one
               DataSet containing the conformations to batch
               conf_ids: the indices into the conformations tensor that consitute this set
               batch_size: size of batches

               iterator returning a dict with
               'batch_ang_neighbor_map': i_idx, j_idx, dist_ij
                   keyed by number of neighbors around one atom (angular cutoff)
                   containing the distance ij for all atom pairs with indices i_idx and j_idx
                   i_idx and j_idx have correct dimension to be broadcasted to the full list of ij pairs by something like:
                      torch.stack(torch.broadcast_tensors(i_idx,j_idx.long()),dim=-1)
                'rad_dist_map'['atom_ij_idx']: tnesorbuffer with 2 columns
                    indexes of center ataom and indexes of i atoms of pairs with cutoff from radialNet
                'rad_dist_map'['dist_ij']: distances of ij atoms

        """

        self.data_set = data_set
        self.epoch = 0
        self.nCPU = nCPU
        self.device = device

        self.n_confs = conf_ids.shape[0]
        if drop_last:
            self.n_confs = self.n_confs // batch_size * batch_size
        self.n_batch = (self.n_confs-1) // batch_size + 1
        self.current_batch = 0 # only used if ncPU==1
        self.loader = ThreadedLoader(data_set, conf_ids, batch_size, self.n_batch, set_type, device)
        self.loader.start(nCPU)

        self.set_type = set_type

    def setEpoch(self, epoch):
        """ Also resets the iterator to the start """
        self.epoch = epoch

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    # noinspection PyShadowingBuiltins
    def __exit__(self, type, value, traceback):
        pass

    def __len__(self):
        return self.n_batch

    def __next__(self):
        if self.nCPU == 1:
            self.current_batch = self.loader.load_one(1,self.current_batch)
            if self.current_batch == 0: self.loader.signal_epoch_end()

        batch = self.loader.queue.get()
        if batch is None:
            raise StopIteration
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"{self.set_type} Batch fetched at {datetime.now()}")

        neigh_map = batch['batch_ang_neighbor_map']
        device_map = {}
        for k,(i,j,d) in neigh_map.items():
            device_map[k]= (i.to(self.device, non_blocking=True),
                            j.to(self.device, non_blocking=True),
                            d.to(self.device, non_blocking=True))
        batch['batch_ang_neighbor_map'] = device_map

        batch_dist_map = batch['batch_dist_map']
        aij = batch_dist_map['batch_atom_ij_idx']
        dij = batch_dist_map['batch_dist_ij']
        device_map = {'batch_atom_ij_idx': aij.to(self.device, non_blocking=True),
                      'batch_dist_ij': dij.to(self.device, dtype=torch.float32, non_blocking=True)}

        batch['batch_dist_map'] = device_map

        return batch


class ThreadedLoader:
    """ Load data in multiple threads to keep up with GPU """
    def __init__(self, data_set:DataSet, conf_ids:torch.Tensor,
                 batch_size:int, n_batch:int, set_type:str, device):
        """
           padding_map: map(k,v) suggesting that neighbor lists of k neighbors be padded to v
        """

        self.data_set  = data_set
        self.batch_size= batch_size
        self.n_batch   = n_batch
        self.conf_ids  = conf_ids
        self.n_confs   = conf_ids.shape[0]
        self.padding_map = data_set.conf.get('padding', {}).get('ang_neighbors', {})
        self.ang_cutoff  = data_set.conf.get('angleNet',{}).get('angularCutoff', 9e9)
        self.is_cuda   = 'cuda' in str(device)

        for frm,too in self.padding_map.items():
            assert frm < too, f"frm < too error {frm}, {too}"
            assert too not in self.padding_map.keys()

        self.set_type = set_type
        self.do_shuffle = set_type == 'train'
        if self.do_shuffle: self._shuffle()

        self.queue:queue.Queue
        self.epoch_barrier:Barrier

        self.terminate = False
        self.threads:List[threading.Thread] = []

    def start(self, nWorker:int):
        assert nWorker >= 1
        self.queue = queue.Queue(nWorker)
        if nWorker > 1:
            self.epoch_barrier = Barrier(nWorker, action=self.signal_epoch_end)
            for i in range(nWorker):
                t = threading.Thread(target=self.loader, args=(nWorker, i,), daemon=True)
                t.start()
                self.threads.append(t)

    def stop(self):
        self.terminate = True

        # noinspection PyShadowingNames
        def _clear_queue(self):
            while True:
                self.queue.get()

        t = threading.Thread(target=_clear_queue, args=(self,), daemon=True)
        t.start()

        for t in self.threads: t.join()


    def _shuffle(self):
        """ it is not safe to calll this will the loader thread is running """
        perm = torch.randperm(self.n_confs)
        self.conf_ids = self.conf_ids[perm]


    def signal_epoch_end(self):
        self.queue.put(None)
        if self.do_shuffle: self._shuffle()


    def load_one(self, nParts:int, current_batch):
        """ will add one batch to self.queue
            after the last batch is added and additional None will be added
            then the conf_ids are shuffled and the next call will add a new batch to the queue
            nParts: total number or worker threads
        """
        ds = self.data_set

        if current_batch >= self.n_batch:
            if nParts == 1:
                return 0 # no threading
            else:
                self.epoch_barrier.wait()
                return current_batch % nParts

        start = current_batch * self.batch_size
        batch_conf_ids = self.conf_ids[start: start+self.batch_size]

        self.is_batch_conf = torch.zeros(self.data_set.n_confs, dtype=torch.bool)
        self.is_batch_conf[batch_conf_ids] = 1

        is_batch_atom = self.is_batch_conf[ds.atoms_long.buffer[:,0]]

        # create tensor for angular calculation
        batch_ang_neighbor_map:Dict[int,Collection[List[torch.tensor]]]  = {}
        for nNeigh, (i_buffer, j_buffer, d_buffer) in ds.ang_neighbor_map.items():
            i_idx = i_buffer.buffer.long()
            dist_ij = d_buffer.buffer

            is_batch_neighbor= is_batch_atom[i_idx.flatten()]
            dist_ij = dist_ij[is_batch_neighbor]

            if dist_ij.shape[0] == 0: continue  # no atoms for this batch

            j_idx= j_buffer.buffer
            i_idx= i_idx[is_batch_neighbor]
            j_idx= j_idx[is_batch_neighbor]

            if nNeigh in self.padding_map:
                to_nNeigh = self.padding_map[nNeigh]
                n_pad = to_nNeigh-nNeigh
                j_idx   = torch.cat((j_idx,i_idx.int().expand(-1,n_pad)),dim=-1)
                dist_ij = torch.cat((dist_ij,
                                     dist_ij.new_full((dist_ij.shape[0],n_pad), self.ang_cutoff)),
                                     dim=-1)
                nNeigh = to_nNeigh

            if nNeigh not in batch_ang_neighbor_map:
                batch_ang_neighbor_map[nNeigh] = ([],[],[])
            i_list,j_list, d_list = batch_ang_neighbor_map[nNeigh]
            i_list.append(i_idx)
            j_list.append(j_idx)
            d_list.append(dist_ij)

        for (nNeigh, (i_idx_list, j_idx_list, dist_ij_list)) in batch_ang_neighbor_map.items():
            i_idx = torch.cat(i_idx_list, 0)
            j_idx = torch.cat(j_idx_list, 0)
            dist_ij = torch.cat(dist_ij_list, 0)

            if self.is_cuda:
                batch_ang_neighbor_map[nNeigh]  = (
                    i_idx.to(dtype=torch.long).contiguous().pin_memory(),
                    j_idx.to(dtype=torch.long).contiguous().pin_memory(),
                    dist_ij.to(dtype=torch.float).contiguous().pin_memory())
            else:
                batch_ang_neighbor_map[nNeigh]  = (
                    i_idx.to(dtype=torch.long),
                    j_idx.to(dtype=torch.long),
                    dist_ij.to(dtype=torch.float))

        # create tensor for radial calculation
        is_rad_atom = is_batch_atom[ds.rad_dist_map['atom_ij_idx'].buffer[:,0]]
        batch_ij_atom = ds.rad_dist_map['atom_ij_idx'].buffer[is_rad_atom]
        batch_dist_ij = ds.rad_dist_map['dist_ij'].buffer[is_rad_atom]
        if self.is_cuda:
            batch_ij_atom = batch_ij_atom.contiguous().pin_memory()
            batch_dist_ij = batch_dist_ij.contiguous().pin_memory()

        batch_dist_map = {'batch_atom_ij_idx': batch_ij_atom,
                          'batch_dist_ij':     batch_dist_ij}

        self.queue.put({'batch_ang_neighbor_map': batch_ang_neighbor_map,
                        'batch_dist_map':         batch_dist_map })
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"{self.set_type} Batch {current_batch} queued at {datetime.now()}")

        current_batch += nParts

        return current_batch


    def loader(self, nparts:int, part:int):
        """ will add one batch after the other to self.queue
            Every time the last batch is added and additional None will be added
        """
        current_batch = part
        while not self.terminate:
            current_batch = self.load_one(nparts, current_batch)

        log.debug(f"Part {part} terminated")





class BatchDataLoader:
    """ Batch of data for DistNEt
    """

    def __init__(self, data_set, batch_ids:torch.Tensor, set_type='train'):
        """
           Argument:
               set_type: if == 'train' than random sampling will be one
               DataSet containing the conformations to batch
               conf_ids: the indices into the conformations tensor that consitute this set
               batch_size: size of batches

               iterator returning a dict with
               'batch_ang_neighbor_map': i_idx, j_idx, dist_ij
                   keyed by number of neighbors around one atom (angular cutoff)
                   containing the distance ij for all atom pairs with indices i_idx and j_idx
                   i_idx and j_idx have correct dimension to be broadcasted to the full list of ij pairs by something like:
                      torch.stack(torch.broadcast_tensors(i_idx,j_idx.long()),dim=-1)
        """

        self.data_set = data_set
        self.batch_ids = batch_ids
        self.n_batch_ids = batch_ids.shape[0]
        self.epoch = 0

        self.current_batch = 0 # only used if ncPU==1
        self.set_type = set_type
        self._do_shuffle = 'train' in set_type
        log.info(f'BatchIDDataLoader created {set_type} nbatch={batch_ids.shape[0]}')


    def _shuffle(self):
        """ it is not safe to call this while the loader thread is running """
        if self._do_shuffle:
            perm = torch.randperm(self.n_batch_ids)
            self.batch_ids = self.batch_ids[perm]


    def setEpoch(self, epoch):
        """ Also resets the iterator to the start """
        self.epoch = epoch
        self.current_batch = 0

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        pass

    def __len__(self):
        return self.n_batch_ids

    def __next__(self):
        if self.current_batch >= self.n_batch_ids:
            if self._do_shuffle: self._shuffle()
            raise StopIteration

        batch = self.data_set.batches[self.batch_ids[self.current_batch]]
        self.current_batch += 1

        return batch

    def set_loss(self, loss):
        pass


class WorstBatchDataLoader(BatchDataLoader):
    """This BatchDataLoader will supress batches with low average loss so that
       the training can focus on high loss batches
    """

    def __init__(self, data_set, batch_ids:torch.Tensor, set_type='train',
                 fract_returned:float = 0.5, full_data_freq:int = 3):
        super().__init__(data_set, batch_ids, set_type)
        device=data_set.device_data_set_list[0].device
        self.last_known_loss = torch.full((len(self),), float("INF"),
                                           dtype=torch.float, device=device)
        self.all_batches = batch_ids.clone()
        self.fract_returned = fract_returned
        self.full_data_freq = full_data_freq

    def set_loss(self, loss):
        self.last_known_loss[self.current_batch-1] = loss

    def _shuffle(self):
        if not self._do_shuffle: return

        if (self.epoch+1) % self.full_data_freq == 0:
            self.batch_ids = self.all_batches
            self.n_batch_ids = self.all_batches.shape[0]
        else:
            self.last_known_loss, idx = self.last_known_loss.sort(descending=True)
            self.all_batches = self.all_batches[idx]
            self.n_batch_ids = int(self.fract_returned*self.all_batches.shape[0]+1)
            log.warning(f"Dataloader epoch {self.epoch+1} restricting to {self.n_batch_ids} batches")
            self.batch_ids = self.all_batches[0:self.n_batch_ids]


class BatchIDDataLoader:
    """
        Return just a list of bat_idx's that refer to the BatchDataSet.batches[]

        This is to be used in DataParallel GPU parallelization.
    """

    def __init__(self, idxs_per_iteration:int, batch_ids:torch.Tensor, set_type='train'):
        """
           Argument:
               idxs_per_iteration: must be equal to the number of GPU used
               set_type: if == 'train' than random sampling will be on
               batch_ids: the indices into the DataSet.batches list that stores batches for this loader

               iterator returning a tensor with idxs_per_iteration indexes
        """

        self.batch_ids = batch_ids
        self.n_batch_ids = batch_ids.shape[0]
        self.idxs_per_iteration = idxs_per_iteration
        self.epoch = 0

        self.current_idx = 0
        self.set_type = set_type
        self._do_shuffle = 'train' in set_type

        log.info(f'BatchIDDataLoader created {set_type} nbatch={batch_ids.size()}')

    def _shuffle(self):
        """ it is not safe to calll this will the loader thread is running """
        perm = torch.randperm(self.n_batch_ids)
        self.batch_ids = self.batch_ids[perm]


    def setEpoch(self, epoch):
        """ Also resets the iterator to the start """
        self.epoch = epoch
        self.current_idx = 0

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        pass

    def __len__(self):
        return self.n_batch_ids

    def __next__(self):
        if self.current_idx >= self.n_batch_ids:
            if self._do_shuffle: self._shuffle()
            raise StopIteration

        batch = self.batch_ids[self.current_idx : self.current_idx+self.idxs_per_iteration]
        #warn(f'dl {batch} {self.current_idx} {self.n_batch_ids} {self.batch_ids.shape}')
        self.current_idx += self.idxs_per_iteration

        return batch

    def set_loss(self, loss):
        pass
