#!/usr/bin/env python


import torch

import logging
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611
import pickle
import gzip
from ml_qm.distNet.data_loader import ThreadedLoader

from ml_qm.distNet.data_set import DataSet, AtomTypes, DeviceDataSet
from cdd_chem.util import constants
import collections
import time
import gc
from typing import IO, cast, Dict, Any


log = logging.getLogger(__name__)


def _pin_memory(obj):
    if isinstance(obj, (list, tuple, set)):
        ret = type(obj) (_pin_memory(x) for x in obj)
        return ret

    elif isinstance(obj, dict):
        return type(obj) ((k, _pin_memory(v)) for k, v in obj.items())

    else:
        newo = obj.contiguous().pin_memory()
        del obj
        return newo

#     else:
#         warn(f'{type(obj)}')
#         warn(f'   {obj.device}')
#         warn(f'      {obj.is_pinned()}')
#         try:
#             newo = obj.pin_memory()
#         except RuntimeError:
#             warn(obj)
#         del obj
#         return newo
#

class BatchDataSet():
    """ Dataset that loads all batches of conformations into main memory """

    def __init__(self, conf, pickle_file, skip_factor:int = 1):
        '''
           Argument:
               skip_factor: only load n_batches / skipFactor batches
        '''
        self.device_data_set_list = []
        with gzip.open(pickle_file,'rb') as infile:
            dds = pickle.load(cast(IO[bytes], infile))

            #allow changes to atom type definitions
            dds.atom_types = AtomTypes(conf['atom_types'], conf['atom_embedding'])
            self.device_data_set_list.append(dds)

            self.batches = []

            info = pickle.load(cast(IO[bytes], infile))
            self.n_confs = info['n_confs']
            self.n_batch = info['n_batch']
            log.info(f"Reading {pickle_file} n_conf={self.n_confs} n_batch={self.n_batch}")

            load_batches = 999999999999
            if skip_factor > 1:
                load_batches = self.n_batch//skip_factor
                self.n_batch = load_batches

            b = pickle.load(cast(IO[bytes], infile))
            while b and load_batches > 0:
                self.batches.append(b)
                b = pickle.load(cast(IO[bytes], infile))
                load_batches -= 1

            assert self.n_batch == len(self.batches)

        self.batch_size = conf['batchSize']

    def get_device_data_set(self, device=None) -> DeviceDataSet:
        if device is None or device.index is None:
            return self.device_data_set_list[0]

        return self.device_data_set_list[device.index]


    def to_(self, devices):
        ''' move device_data_set to devices
            pin other batch info if device == cuda

            Arguments:
            devices: device of list of device ids to move device_dataset too
        '''

        if isinstance(devices, collections.abc.Sequence) and not isinstance(devices, str):
            mainDev = devices.pop(0)
        else:
            mainDev = devices
            devices = []

        self.device_data_set_list[0].to_(mainDev)
        for dev in devices:
            self.device_data_set_list.append(self.device_data_set_list[0].to(dev))

        if 'cuda' in str(mainDev) or isinstance(mainDev, int):
            self.batches = _pin_memory(self.batches)
            gc.collect()



    @staticmethod
    def savePickle(data_set:DataSet, pickle_file:str,
                 batch_size:int, drop_last:bool = False,
                 shuffle:bool = True, nCPU:int = 2):
        '''
        Arguments:
            conf_ids list of tensors containing the conformation ids for each set
            set_type: type of each set 'train'|'val'|'test'
        '''

        n_confs = data_set.n_confs
        if shuffle:
            conf_ids = torch.randperm(n_confs)
        else:
            conf_ids = torch.arange(n_confs)

        if drop_last:
            n_confs = n_confs // batch_size * batch_size
            conf_ids = conf_ids[0:n_confs]
        n_batch = (n_confs-1) // batch_size + 1

        log.info(f"Writing {pickle_file} batch_size={batch_size} n_conf={n_confs} n_batches={n_batch}")
        with gzip.open(pickle_file, mode='wb') as out:
            device_data_set = data_set.to('cpu')
            pickle.dump(device_data_set, cast(IO[bytes], out))

            loader = ThreadedLoader(data_set, conf_ids, batch_size, n_batch, 'val', 'cpu')
            if nCPU <= 1: nCPU = 2  ## this currently only works multi threaded
            loader.start(nCPU)

            pickle.dump({'n_confs': n_confs,
                         'n_batch': n_batch   }, cast(IO[bytes], out))

            c_batch = 0
            b: Dict[str,Any]
            for b in iter(loader.queue.get, None):
                pickle.dump(b, cast(IO[bytes], out))
                c_batch += 1
                if c_batch % (n_batch // 10) == 0:
                    log.info(f"Done with batch {c_batch}/{n_batch}")
            pickle.dump(None, cast(IO[bytes], out))

        log.info(f"Completed writing {pickle_file} number of batches: {n_batch}")
        assert c_batch == n_batch
        loader.stop()



def main(argv=None): # IGNORE:C0111
    import argparse
    import yaml
    from cdd_chem.util import log_helper


    #mp.set_start_method('spawn'); warn("spawn")

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=
        "This will create two pickle files for use in training Dist NNP models:"
        "  as defined by pickleFile and batchPickleFile containing the DataSet and the "
        "  batched data set. These act as caches and need to be deleted if the program fails "
        "  or you change the information for their creation in the yml file")

    parser.add_argument('-yaml' ,  metavar='filename' ,  type=str, dest='yml', required=True,
                        help='input file')

    parser.add_argument('-nCPU' ,  metavar='n' ,  type=int, default=4,
                        help='number of CPUs used for creating the batched data set (def:4)')

    parser.add_argument('-logINI',  metavar='fileName',dest='logFile', type=str,
                        help='ini file for logging (existing file or one of silent,debug)')

    parser.add_argument('-dataSetOnly',  action='store_true', default=False,
                        help='If given the batched data set will not be created')

    parser.add_argument('-nGPU' ,  metavar='n' ,  type=int, default=0,
                        help='number of GPUs, (currently only one supported)')


    args = parser.parse_args()

    log_helper.initialize_loggger(__name__, args.logFile)

    torch.set_num_threads(4)
    device = torch.device("cpu")

    if args.nGPU > 0:
        for i in range(0,10):
            if not torch.cuda.is_available():
                log.warning("GPU not available waiting 5 sec")
                time.sleep(5)
            else:
                device = torch.device("cuda")
                break

    log.info("Reading conf from %s" % args.yml)
    with open(args.yml) as yFile:
        conf = yaml.safe_load(yFile)

    data_set = DataSet(conf, device)
    data_set.load_conf_data()
    data_set.finalize()

    if args.dataSetOnly: return

    tconf = conf['trainData']
    bsize = conf['batchSize']
    bpickle = tconf['batchPickleFile']
    bpickle = f"{bpickle}.{bsize}.pickle.gz"
    bpickle = constants.replace_consts_and_env(bpickle)
    BatchDataSet.savePickle(data_set, bpickle,  bsize,
                              tconf.get('batch.dropLast'), True, args.nCPU)

#    bd = BatchDataSet(bpickle)
#    bd.to_('cpu')


TESTRUN = 0
PROFILE = 0

if __name__ == "__main__":
    import sys


    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE == 1:
        import cProfile
        import pstats
        profile_filename = 'ml_qm.optimize.optimizer_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = sys.stderr
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    elif PROFILE == 2:
        import line_profiler
        import atexit
        from ml_qm.distNet.dist_net import ComputeRadialInput, AngleNet

        prof = line_profiler.LineProfiler()
        atexit.register(prof.print_stats)

        prof.add_function(ComputeRadialInput.forward)
        prof.add_function(AngleNet.forward)
#         prof.add_function(optmz.Froot.evaluate)
#         prof.add_function(optmz.trust_step)
#         prof.add_function(optmz.getCartesianNorm)
#         prof.add_function(optmz.get_delta_prime)
#         prof.add_function(internal.InternalCoordinates.newCartesian)

        prof_wrapper = prof(main)
        prof_wrapper()
        sys.exit(0)

    elif PROFILE == 3:
        from pyinstrument import Profiler
        profiler = Profiler()
        profiler.start()

        main()

        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
        sys.exit(0)

    elif PROFILE == 4:
        # this needs lots of memory
        import torch.autograd
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            main()
        # use chrome://tracing  to open file
        prof.export_chrome_trace("profile_Chrome.json")
        sys.exit(0)

    sys.exit(main())
