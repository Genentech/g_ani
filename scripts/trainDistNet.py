#!/bin/env python
import os
import sys
import time
import argparse
import glob
import gc
import torch.nn as nn
from ml_qm.nn.util import createOptimizer
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import mse_loss
import torch.nn.utils as tutils

import torch.cuda
import torch.autograd
import numpy as np
import yaml
from typing import Dict
import logging

from cdd_chem.util import log_helper
from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611
import cdd_chem.util.debug.memory
from cdd_chem.util import constants
from ml_qm.distNet.dist_net import EnergyNet, ParallelEnergyNet
from ml_qm.pt.nn.CheckPoint import CheckPointer
from ml_qm.distNet.data_loader import BatchDataLoader, BatchIDDataLoader
from ml_qm.distNet.batch_data_set import BatchDataSet
from ml_qm.pt.ranger import Ranger
from ml_qm.pt.scheduler import create_scheduler
from ml_qm.pt.nn.loss import createLossFunction



log = logging.getLogger(__name__)
INF = float("inf")
TESTRUN = 0
PROFILE = 0


def printWeights(model, prefix=None):
    if prefix is not None: sys.stderr.write(prefix)

    for name, weights in model.named_parameters():
        warn("%s: %s" % (name,weights))
    warn("-"*30)




def createOptimizerOld(model, conf):
    optParam = [ { 'params' : model.parameters(),  **conf['optParam'] } ]

    if conf['optType'] == 'Ranger':
        optimizer = Ranger
    else:
        optimizer = getattr(optim, conf['optType'], None)
    if optimizer is not None:
        optimizer = optimizer(optParam)
    else:
        raise TypeError("unknown optimizer: " + conf['optType'])

    return optimizer


def split_batches(conf:Dict[str,object], n_batch:int):
    tconf = conf['trainData']
    fractions = np.array([tconf['trainFraction'], tconf['valFraction'], tconf['testFraction']])
    fractions = fractions / fractions.sum() # normalize
    frac_count = (n_batch * fractions).astype(int)
    frac_count[0] = n_batch - sum(frac_count[1:])

    # make sure we always split the same way
    master_rng_state = torch.get_rng_state()
    torch.manual_seed(conf['trainData']['seed'])
    batch_ids = torch.randperm(n_batch)
    torch.set_rng_state(master_rng_state)

    return batch_ids.split(frac_count.tolist())


#@profile
def train_model(data_set:BatchDataSet, model:EnergyNet, checkPointer:CheckPointer,
                device, dataloaders:Dict[str,BatchDataLoader], loss_fn, optimizer,
                scheduler=None, gradClipping:int = None,
                num_epochs=25, maxNoImprovementEpochs:int = 1000, bohb_batch_fraction:float = 1,
                minMSEValidation=float("inf") ):
    """
        num_epochs: stop after numEpoch epochs
        minMSEValidation: run validation only if training MSE is smaller
    """

    device_data_set = data_set.get_device_data_set()

    minMaxMSE = 5
    trainLossFuncIsMSE = True
    if not isinstance(loss_fn,nn.MSELoss):
        trainLossFuncIsMSE = False

    trainMSE  = 9e9999
    valMSE    = 9e9999
    minValMSE = 9e9999
    maxMSE    = 9e9999
    noImprovementEpochs = 0

    phases = ['train']
    if 'val'  in dataloaders: phases.append('val')
    if 'test' in dataloaders: phases.append('test')

    for epoch in range(checkPointer.numEpoch, num_epochs):
        if log.isEnabledFor(logging.DEBUG):
            if "cuda" in str(device):
                torch.cuda.empty_cache()
                log.debug(f"Before epoch {epoch}: mem allocated: {torch.cuda.memory_allocated(device)} "
                          f"mem cached: {torch.cuda.memory_reserved(device)} "
                          f"max allocated: {torch.cuda.max_memory_allocated(device)} "
                          f"max mem cached:{torch.cuda.max_memory_reserved(device)}")
            log.debug(f"CPU RSS(gb): {cdd_chem.util.debug.memory.get_rss_gb()}")

        testMSE = None
        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
                usedLossFn = loss_fn

            else:  # phase == val or test
                model.eval()   # Set model to evaluate mode
                usedLossFn = nn.MSELoss()

                if phase == 'test':
                    maxMSE = valMSE

                    if valMSE < trainMSE: maxMSE = trainMSE

                    if maxMSE > minMaxMSE and \
                      (valMSE > minValMSE * 1.1 or maxMSE > minMaxMSE * 1.2 ):
                        break

                    log.info("maxMSE=%.3f minMaxMSE=%.3f  valMSE=%.3f minValMSE=%.3f"
                         % (maxMSE, minMaxMSE,  valMSE, minValMSE))
                    minMaxMSE = maxMSE


            sum_sq_err  = torch.zeros(1, device=device)
            sum_loss = torch.zeros(1, device=device)
            sum_cnt  = torch.zeros(1, device=device)

            # Iterate over data.
            dl = dataloaders[phase]
            dl.setEpoch(epoch)
            nbatch = len(dl)
            if bohb_batch_fraction < 1:
                nbatch = int(nbatch * bohb_batch_fraction + 1)

            for i, batch in enumerate(dl):
                starte = time.perf_counter()
                if i >= nbatch: break

                batch_start = time.time()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    startp = time.perf_counter()
                    pred = model(data_set, batch)
                    if log.isEnabledFor(logging.DEBUG):
                        log.debug(f'{phase}: pred time={time.perf_counter()-startp}')

                    conf_idx = pred['batch_output_conf_idx']
                    e        = pred['batch_output']
                    expected = device_data_set.conformations[conf_idx,0]

                    del batch
                    loss = usedLossFn(e, expected)
                    #if log.isEnabledFor(logging.DEBUG):
                    #    log.debug(f'{phase} loss={loss.cpu()}')

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if getattr(scheduler,'batch_step', False):
                            scheduler.batch_step(epoch + i/nbatch)

                        startb = time.perf_counter()
                        loss.backward()
                        if log.isEnabledFor(logging.DEBUG):
                            log.debug(f'backward time={time.perf_counter()-startb}')

                        if gradClipping:
                            # note clip_grad_norm converts to CPU and slows down training by 5%
                            tutils.clip_grad_norm_(model.parameters(), gradClipping)
                        optimizer.step()

                        loss.detach_()
                        loss.requires_grad_(False)
                        dl.set_loss(loss)
                        mse = loss
                        if not trainLossFuncIsMSE:
                            sum_loss += loss.mul_(expected.size(0))
                            e.detach_()
                            mse = mse_loss(e,expected)
                    else:
                        mse = loss

                    del e, pred

                # statistics, loss is detached
                sum_sq_err  += mse.mul_(expected.size(0))
                sum_cnt  += expected.size(0)
                del mse, loss, expected

                #log.debug(f"Batch end {time.time() - batch_start:.6f} sec")


            epochMSE = sum_sq_err / sum_cnt

            if phase == 'train':
                trainMSE = epochMSE.cpu().item()
                #checkPointer.printTrainResults(trainMSE)

                if not trainLossFuncIsMSE:
                    epochTrainLoss = (sum_loss / sum_cnt).cpu().item()
                    log.info("Epoch %3s TrainLoss: %-7.3f" % (checkPointer.numEpoch, epochTrainLoss))

            elif phase == 'val':
                valMSE = epochMSE.cpu().item()
                if valMSE <  minValMSE:
                    minValMSE = valMSE
                    noImprovementEpochs = 0
                else:
                    noImprovementEpochs += 1
            else:  # test
                testMSE = epochMSE.cpu().item()

            del sum_sq_err, sum_cnt, sum_loss, epochMSE

        checkPointer.checkPoint(trainMSE, valMSE)
        checkPointer.printResults(trainMSE, valMSE, testMSE)
        checkPointer.incrementEpoch()
        for _, dl in dataloaders.items(): dl.setEpoch(checkPointer.numEpoch)
        if noImprovementEpochs >= maxNoImprovementEpochs: break

        # as of pytorch 1.1 scheduler.step should be called after opt.step
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                if valMSE < INF: scheduler.step(valMSE)
            else:
                scheduler.step()

        if log.isEnabledFor(logging.DEBUG):
            log.debug(f'epoch time={time.perf_counter()-starte}')



    print('Training completed')
    print('Best Val MSE: %.4f' % (checkPointer.bestValLoss))

    checkPointer.saveCheckPoint()

    return checkPointer.numEpoch, model


def load_model(model, model_file:str = None):
    if not model_file: return model

    log.info("loading model from: %s" % model_file)
    checkpoint = torch.load(model_file, map_location='cpu')
    model.load_state_dict(checkpoint['model'], False)

    return model





def load_data_set(conf) -> BatchDataSet:
    tconf = conf['trainData']
    skip_factor = tconf.get("skipFactor", 1)
    pickleFile = tconf['batchPickleFile']
    batch_pickle = f"{pickleFile}.{conf['batchSize']}.pickle.gz"
    batch_pickle = constants.replace_consts_and_env(batch_pickle)
    data_set = BatchDataSet(conf, batch_pickle, skip_factor)
    return data_set


def createModel(conf, data_set:BatchDataSet, device, modelFile:str = None):
    train_batch, val_batch, test_batch = split_batches(conf, data_set.n_batch)

    #clean up memory hoping to pack better
    gc.collect(); gc.collect()

    if torch.cuda.device_count() > 1:
        data_set.to_([i for i in range(torch.cuda.device_count())])
        ngpu = torch.cuda.device_count()
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(ParallelEnergyNet(conf))
        dataLoaders = [BatchIDDataLoader(ngpu, train_batch, 'train'),
                       BatchIDDataLoader(ngpu, val_batch, 'val'),
                       BatchIDDataLoader(ngpu, test_batch, 'test')]
    else:
        data_set.to_(device)
        model = EnergyNet(conf)
        dataLoaders = [BatchDataLoader(data_set, train_batch, 'train'),
                       BatchDataLoader(data_set, val_batch, 'val'),
                       BatchDataLoader(data_set, test_batch, 'test')]

    load_model(model, modelFile)
    model.to(device)

    dataLoaders = [d for d in dataLoaders if len(d) > 0]
    dataLoaderMap = {}
    for dl in dataLoaders:
        if dl.set_type in dataLoaderMap:
            raise TypeError(f"Dataloader of type: '{dl.set_type} defined twice")
        dataLoaderMap[dl.set_type] = dl

    return model, data_set, dataLoaderMap


def main(argv=None): # IGNORE:C0111


    #mp.set_start_method('spawn'); warn("spawn")

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="")

    parser.add_argument('-yaml', metavar='filename' ,  type=str, dest='yml', required=True,
                        help='input file')

    parser.add_argument('-nGPU', metavar='n', type=int, default=0,
                        help='number of GPUs, (currently only one supported)')

    parser.add_argument('-nCPU', metavar='n', type=int, default=4,
                        help='number of CPUs used for data loading (def:4)')

    parser.add_argument('-deleteCheckPoints', action='store_true', default=False,
                        help='Delete pre-existing checkpoint files on startup')

    parser.add_argument('-minMSEValidation', type=float, default=9e999,
                        help='Do not run validation unless MSE is below this value')

    parser.add_argument('-loadModel', metavar='filename' ,  type=str, dest='modelFile',
                        help='file with model weights to load to initialize model instead of using random weights.' +
                             'This can be either a "chkpty_" file or a "best_" file')

    parser.add_argument('-logINI',  metavar='fileName',dest='logFile', type=str,
                        help='ini file for logging (existing file or one of silent,debug)')

    args = parser.parse_args()

    log_helper.initialize_loggger(__name__, args.logFile)

    # needed if loading data onto cuda in dataloader
    #
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    torch.set_num_threads(4)
    torch.cuda.empty_cache()
    device = torch.device("cpu")
    if args.nGPU > 0:
        for _ in range(0,10):
            if not torch.cuda.is_available():
                log.warning("GPU not available waiting 5 sec")
                time.sleep(5)
            else:
                device = torch.device("cuda")
                break

    devId = ""
    if str(device) == "cuda":
        devId = torch.cuda.current_device()
    log.info("nGPU=%i Device=%s %s" % (args.nGPU,device,devId))

    log.info("Reading conf from %s" % args.yml)
    with open(args.yml) as yFile:
        conf = yaml.safe_load(yFile)

    if args.deleteCheckPoints:
        for f in glob.glob("checkpoints/chkpty_*"):
            os.remove(f)

    if args.modelFile is not None:
        if len(glob.glob('checkpoints/chkpty_*')) != 0:
            log.critical( "Checkpoint directory is not empty, This conflicts with -loadModel. Please delete.")
            sys.exit(1)

    lossF = createLossFunction(conf)

    batch_data_Set = load_data_set(conf)

    gc.collect(); gc.collect()
    model, data_set, dataLoaderMap = createModel(conf, batch_data_Set, device, args.modelFile)

    optimizer = createOptimizer(model, conf)
    scheduler = create_scheduler(optimizer, conf)
    gradClipping = conf.get('gradientClippingMaxNorm', None)


    warn(model)

    maxEpoch = conf['epochs']
    min_save_val = conf.get('minValSave', 2.5)
    checkPointer = CheckPointer(model, optimizer, scheduler,  device, min_save_val)
    _ = checkPointer.restoreLast()

    maxNoImprovementEpochs = conf.get('maxNoImprovementEpochs',200)
    # here is the real work

    start_time = time.time()
    _, model = train_model(data_set, model, checkPointer, device, dataLoaderMap, lossF,
                        optimizer, scheduler, gradClipping=gradClipping, num_epochs=maxEpoch,
                        maxNoImprovementEpochs=maxNoImprovementEpochs,
                        minMSEValidation=args.minMSEValidation  )
    warn(f"Training completed after {time.time() - start_time:.6f} sec")



if __name__ == "__main__":
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
        import torch
        with torch.autograd.profiler.profile(enabled=True,
                                             use_cuda=True,
                                             record_shapes=False) as prof:
            warn("Profiling")
            main()
        print(prof.key_averages().table(sort_by="cuda_time_total"))
        # use chrome://tracing  to open file
        warn("writing profile_Chrome.json")
        prof.export_chrome_trace("profile_Chrome.json")
        sys.exit(0)

    sys.exit(main())
