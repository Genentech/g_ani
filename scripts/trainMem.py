#!/bin/env python
import os
import sys
import time
import json
import argparse
import glob
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import mse_loss

from ml_qm.pt.nn.ani_net import ANINetInitializer
from ml_qm.pt.nn.CheckPoint import DescCompCheckPointer
from ml_qm.pt.nn.regularization import createRegularizer
import ml_qm.pt.nn.loss

from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611
from cdd_chem.util.debug.memory import print_memory_usage
from ml_qm.pt.MEMDataSet import ANIMEMSplitDataSet, ANIMEMSingleDataSet
from ml_qm.pt.MEMBatchDataSet import MEMBatchDataSet, MultiBatchComputerDataSet
from cdd_chem.util import log_helper

import logging
from ml_qm.pt.ranger import Ranger
import ml_qm.pt.nn.Precision_Util as pu
from ml_qm.pt.scheduler import create_scheduler
from ml_qm.pt.nn.loss import createLossFunction
log = logging.getLogger(__name__)


TESTRUN = 0
PROFILE = 0
INF = float("inf")

def printWeights(model, prefix=None):
    if prefix is not None: sys.stderr.write(prefix)

    for name, weights in model.named_parameters():
        warn("%s: %s" % (name,weights))
    warn("-"*30)



def preTrain(jsonFile, modelFile, trainDS, checkPointer, loadSavedDescriptorParam, device):
    """ Pre-train with different dropOutPct and ExPLoss loss function until preRunMSE is reached
    """

    dLoaders = {'train':trainDS}

    # in all maps replace obj[key] with obj["pre." + key] if it exists
    # to allow for preTraining settings
    def replaceWithPreTrainConf(obj):
        for key in obj.keys():
            if "pre." + key in obj:
                obj[key] = obj["pre." + key]
        return obj

    with open(jsonFile) as jFile:
        conf = json.load(jFile, object_hook=replaceWithPreTrainConf)

    ptLossF = createLossFunction(conf)
    ptRegularizer= createRegularizer(conf)
    initAni      = ANINetInitializer(conf,device)
    ptDescComput = initAni.create_descriptor_computer()
    ptModel      = initAni.create_model(modelFile, ptDescComput, loadSavedDescriptorParam)
    ptOptimizer  = createOptimizer(ptModel, ptDescComput, conf)
    ptScheduler  = create_scheduler(ptOptimizer, conf)
    ptMaxEpoch   = conf['epochs']

    ptModel = ptModel.to(device)
    ptDescComput.to(device)
    ptCheckPointer = DescCompCheckPointer(ptModel, ptDescComput, ptOptimizer, ptScheduler, device)

    train_model(ptModel, ptDescComput, ptCheckPointer, device, dLoaders, ptLossF, ptOptimizer,
                ptScheduler, ptRegularizer, num_epochs=ptMaxEpoch, stopTrainMSE=conf['preRunMSE'])

    del ptCheckPointer, ptScheduler, ptOptimizer, ptDescComput, ptModel, ptLossF, ptRegularizer

    # load pre-trained model
    checkPointer.restoreLast()


def createDataLoaders(descComput, conf, device):
    atom_types    = conf['atomTypes']
    bSize         = conf['batchSize']
    dropLast      = conf['batch.dropLast']
    trainDataConf = conf['trainData']
    energy_norm   = conf.get('energyNormalization', 'diff')

    ads = ANIMEMSplitDataSet()
    ads.load_ANI_files(atom_types, trainDataConf)

    print_memory_usage()

    fuzz          = conf.get('fuzzCoordinates', None)
    descBatchSize = conf.get("batchSize.descriptors.mult", 1) * bSize

    # do not drop last batch if we will subbatch
    doprLastMemB = dropLast if descBatchSize == bSize else False

    trainDS       = MEMBatchDataSet(ads, descBatchSize, descComput, 'train', False, doprLastMemB, fuzz, device)
    valDS         = MEMBatchDataSet(ads, descBatchSize, descComput, 'val', False, False, fuzz, device)
    del ads ## free memory

    print_memory_usage()

    testFile = trainDataConf['tFile']
    atds = ANIMEMSingleDataSet( 'test' )
    atds.load_ANI_files(testFile, atom_types, trainDataConf)

    print_memory_usage()

    testDS = MEMBatchDataSet(atds, descBatchSize, descComput, 'test', False, False, None, device)
    del atds ## free memory

    print_memory_usage()

    if descBatchSize > bSize:
        trainDS = MultiBatchComputerDataSet(trainDS, bSize, dropLast)

    gc.collect()
    gc.collect()
    print_memory_usage()

    return trainDS, valDS, testDS


def createOptimizer(model, descComput, conf):
    optParam = [ { 'params' : model.parameters(),  **conf['optParam'] } ]
    if len(descComput.radialBasis.getOptParameter()) > 0:
        oparams = conf.get('radialBasisOptParam', [])
        if isinstance(oparams,dict):
            # use same parameters for all optimization groups
            for computP in descComput.radialBasis.getOptParameter():
                optParam.append( { 'params' : computP, **oparams })
        else:
            computParams = descComput.radialBasis.getOptParameter()
            if len(computParams) != len(oparams):
                raise Exception("Len of radialBasisOptParam does not equal number of groups %s != %s"
                                % (len(computParams), len(oparams)))
            # assume param is list of dicts with individual configurations per optimization group
            for computP, optP in zip(computParams, oparams):
                optParam.append( { 'params' : computP, **optP })

    oparams = conf.get('angularOptParam', [])
    if isinstance(oparams,dict):
        # use same parameters for all optimization groups
        for computP in descComput.angularBasis.getOptParameter():
            optParam.append( { 'params' : computP, **oparams })
    elif descComput.angularBasis is not None:
        computParams = descComput.angularBasis.getOptParameter()
        if len(computParams) != len(oparams):
            raise Exception("Len of angularBasisOptParam does not equal number of groups %s != %s"
                            % (len(computParams), len(oparams)))
        # assume param is list of dicts with individual configurations per optimization group
        for computP, optP in zip(computParams, oparams):
            optParam.append( { 'params' : computP, **optP })


    if conf['optType'] == 'Ranger':
        optimizer = Ranger
    else:
        optimizer = getattr(optim, conf['optType'], None)
    if optimizer is not None:
        descComput.train()
        optimizer = optimizer(optParam)
        descComput.eval()
    else:
        raise TypeError("unknown optimizer: " + conf['optType'])

    return optimizer


#@profile
def train_model(model, descComput, checkPointer, device, dataloaders, loss_fn, optimizer,
                scheduler=None, regularizer=None,
                num_epochs:int = 25, maxNoImprovementEpochs:int = 1000,
                stopTrainMSE:int = 0, minMSEValidation:float = float("inf") ):
    """
        num_epochs: stop after numEpoch epochs
        minMSEValidation: run validation only if training MSE is smaller
    """

    minMaxMSE = 5.
    trainLossFuncIsMSE = True if isinstance(loss_fn,nn.MSELoss) else False

    trainMSE  = 9e9999
    valMSE    = 9e9999
    minValMSE = 9e9999
    maxMSE    = 9e9999
    noImprovementEpochs = 0

    phases = ['train']
    if 'val'  in dataloaders: phases.append('val')
    if 'test' in dataloaders: phases.append('test')

    for epoch in range(checkPointer.numEpoch, num_epochs):
        testMSE = None
        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                usedLossFn = loss_fn

                model.train()  # Set model to training mode
                descComput.train()
            else:  # phase == val or test
                # always use MSE for validation
                usedLossFn = nn.MSELoss()
                if trainMSE > minMSEValidation:
                    break


                if phase == 'val':
                    pass
                else:  # test phase
                    maxMSE = valMSE

                    if valMSE < trainMSE: maxMSE = trainMSE

                    if maxMSE > minMaxMSE and \
                      (valMSE > minValMSE * 1.1 or maxMSE > minMaxMSE * 1.2 ):
                        break

                    minMaxMSE = maxMSE

                model.eval()   # Set model to evaluate mode
                descComput.eval()

            running_MSE  = torch.zeros(1, device=device)
            running_loss = torch.zeros(1, device=device)
            running_cnt  = torch.zeros(1, device=device)

            # Iterate over data.
            nbatch = len(dataloaders[phase])
            i = 0
            for i, batch in enumerate(dataloaders[phase]):
                expected = batch.results

                if False:    # debug
                    batch.printInfo()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    pred = model(batch).to(dtype=pu.NNP_PRECISION.lossDType)
                    del batch
                    loss = usedLossFn(pred, expected)
                    if log.isEnabledFor(logging.DEBUG):
                        log.debug(f'loss={loss.cpu()}')

                    # backward + optimize only if in training phase
                    if phase == 'train':

                        #CombinedScheduler._printLR(optimizer)
                        if getattr(scheduler,'batch_step', False):
                            scheduler.batch_step(epoch + i/nbatch)

                        loss.backward()
                        optimizer.step()

                        #printWeights(model, prefix="-------------------- ")
                        if regularizer is not None:
                            #printWeights(model, prefix="before -------------------- ")
                            regularizer.regularize(model)

                            #printWeights(model, prefix="after -------------------- ")

                        loss.detach_()
                        loss.requires_grad_(False)
                        if not trainLossFuncIsMSE:
                            running_loss += loss.mul_(expected.size(0))
                            pred.detach_()
                            loss = mse_loss(pred, expected)

                    del pred

                # statistics, loss is detached
                running_MSE  += loss.mul_(expected.size(0))
                running_cnt  += expected.size(0)
                del loss, expected

            epochMSE = running_MSE / running_cnt

            if phase == 'train':
                if epoch % 30 == 0: checkPointer.printBasisOptParam()
                trainMSE = epochMSE.cpu().item()
                dataloaders['train'].shuffle()
                if not trainLossFuncIsMSE:
                    epochTrainLoss = (running_loss / running_cnt).cpu().item()
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

            del running_MSE, running_cnt, running_loss

        checkPointer.checkPoint(trainMSE, valMSE)
        checkPointer.printResults(trainMSE, valMSE, testMSE)
        checkPointer.incrementEpoch()
        for _, dl in dataloaders.items(): dl.setEpoch(checkPointer.numEpoch)
        if trainMSE < stopTrainMSE: break
        if noImprovementEpochs >= maxNoImprovementEpochs: break

        # as of pytorch 1.1 scheduler.step should be called after opt.step
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                if valMSE < INF: scheduler.step(valMSE)
                #if maxMSE < INF: scheduler.step(maxMSE)
            else:
                scheduler.step()

    print('Training completed')
    print('Best Val MSE: %.4f' % (checkPointer.bestValMSE))

    checkPointer.saveCheckPoint()

    return checkPointer.numEpoch, model







def main(argv=None): # IGNORE:C0111

    #mp.set_start_method('spawn'); warn("spawn")

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="")

    parser.add_argument('-json' ,  metavar='filename' ,  type=str, dest='jsonFile', required=True,
                        help='input file')

    parser.add_argument('-nCPU' ,  metavar='n' ,  type=int, default=0,
                        help='number of CPUs')

    parser.add_argument('-nGPU' ,  metavar='n' ,  type=int, default=0,
                        help='number of GPUs, (currently only one supported)')

    parser.add_argument('-deleteCheckPoints',  action='store_true', default=False,
                        help='Delete pre-existing checkpoint files on startup')

    parser.add_argument('-minMSEValidation',  type=float, default=9e999,
                        help='Do not run validation unless MSE is below this value')

    parser.add_argument('-loadModel', metavar='filename' ,  type=str, dest='modelFile',
                        help='file with model weights to load to initialize model instead of using random weights.' +
                             'This can be either a "chkpty_" file or a "best_" file')

    parser.add_argument('-logINI',  metavar='fileName',dest='logFile', type=str,
                        help='ini file for logging (existing file or one of silent,debug)')

    parser.add_argument('-loadSavedDescriptorParam', metavar='True|False' ,  type=str,
                          choices=['False', 'True', "true", "false"], default="true",
                          help='if ')

    args = parser.parse_args()

    log_helper.initialize_loggger(__name__, args.logFile)

    # make parsed parameter local variables
    #locals().update(args.__dict__)

    loadSavedDescriptorParam = True if args.loadSavedDescriptorParam.lower() == 'true' else False

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
        for i in range(0,10):
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

    log.info("Reading conf from %s" % args.jsonFile)
    with open(args.jsonFile) as jFile:
        conf = json.load(jFile)

    if args.deleteCheckPoints:
        for f in glob.glob("checkpoints/chkpty_*"):
            os.remove(f)

    if args.modelFile is not None:
        if len(glob.glob('checkpoints/chkpty_*')) != 0:
            log.critical( "Checkpoint directory is not empty, This conflicts with -loadModel. Please delete.")
            sys.exit(1)

    lossF = createLossFunction(conf)

    regularizer = createRegularizer(conf)

    initAni    = ANINetInitializer(conf,device)
    descComput = initAni.create_descriptor_computer()
    model      = initAni.create_model(args.modelFile, descComput, loadSavedDescriptorParam)

    optimizer = createOptimizer(model, descComput, conf)

    scheduler = create_scheduler(optimizer, conf)

    trainDS, valDS, testDS = createDataLoaders(descComput, conf, device)

    model = model.to(device)
    descComput.to(device)

    maxEpoch = conf['epochs']
    min_save_val = conf.get('minValSave', 3)
    checkPointer = DescCompCheckPointer(model, descComput, optimizer, scheduler, device, min_save_val)
    restored = checkPointer.restoreLast()


    log.info("Starting Training (training batches=%i, val batches=%i)" % ( len(trainDS), len(valDS)))

    # pre minimization with Different parameters.
    if not restored and "preRunMSE" in conf and args.modelFile is None:
        preTrain(args.jsonFile, args.modelFile, trainDS, checkPointer, loadSavedDescriptorParam, device)

    dLoaders = {'train': trainDS,
                'val':   valDS,
                'test':  testDS
                }

    maxNoImprovementEpochs = conf.get('maxNoImprovementEpochs',200)
    # here is the real work
    _, model = train_model(model, descComput, checkPointer, device, dLoaders, lossF,
                        optimizer, scheduler, regularizer,
                        num_epochs=maxEpoch, maxNoImprovementEpochs=maxNoImprovementEpochs,
                        minMSEValidation=args.minMSEValidation  )

if __name__ == '__main__':
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
        from pyinstrument import Profiler # noqa: F401; # pylint: disable=W0611
        profiler = Profiler()
        profiler.start()

        main()

        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
        sys.exit(0)

    elif PROFILE == 4:
        # this needs lots of memory
        import torch.autograd
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
