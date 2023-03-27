#!/bin/env python

"""
Worker for Example 5 - PyTorch
==============================

In this example implements a small CNN in PyTorch to train it on MNIST.
The configuration space shows the most common types of hyperparameters and
even contains conditional dependencies.
In this example implements a small CNN in Keras to train it on MNIST.
The configuration space shows the most common types of hyperparameters and
even contains conditional dependencies.

We'll optimise the following hyperparameters:

+-------------------------+----------------+-----------------+------------------------+
| Parameter Name          | Parameter type |  Range/Choices  | Comment                |
+=========================+================+=================+========================+
| Learning rate           |  float         | [1e-6, 1e-2]    | varied logarithmically |
+-------------------------+----------------+-----------------+------------------------+
| Optimizer               | categorical    | {Adam, SGD }    | discrete choice        |
+-------------------------+----------------+-----------------+------------------------+
| SGD momentum            |  float         | [0, 0.99]       | only active if         |
|                         |                |                 | optimizer == SGD       |
+-------------------------+----------------+-----------------+------------------------+
| Number of conv layers   | integer        | [1,3]           | can only take integer  |
|                         |                |                 | values 1, 2, or 3      |
+-------------------------+----------------+-----------------+------------------------+
| Number of filters in    | integer        | [4, 64]         | logarithmically varied |
| the first conf layer    |                |                 | integer values         |
+-------------------------+----------------+-----------------+------------------------+
| Number of filters in    | integer        | [4, 64]         | only active if number  |
| the second conf layer   |                |                 | of layers >= 2         |
+-------------------------+----------------+-----------------+------------------------+
| Number of filters in    | integer        | [4, 64]         | only active if number  |
| the third conf layer    |                |                 | of layers == 3         |
+-------------------------+----------------+-----------------+------------------------+
| Dropout rate            |  float         | [0, 0.9]        | standard continuous    |
|                         |                |                 | parameter              |
+-------------------------+----------------+-----------------+------------------------+
| Number of hidden units  | integer        | [8,256]         | logarithmically varied |
| in fully connected layer|                |                 | integer values         |
+-------------------------+----------------+-----------------+------------------------+

Please refer to the compute method below to see how those are defined using the
ConfigSpace package.
      
The network does not achieve stellar performance when a random configuration is samples,
but a few iterations should yield an accuracy of >90%. To speed up training, only
8192 images are used for training, 1024 for validation.
The purpose is not to achieve state of the art on MNIST, but to show how to use
PyTorch inside HpBandSter, and to demonstrate a more complicated search space.
"""
from hpbandster.core.worker import Worker
import torch
import time
import logging
import json
from typing import Dict, Any
import os
import copy
import argparse

from cdd_chem.util import log_helper

from scripts.trainMem import createOptimizer, createDataLoaders, train_model
from ml_qm.pt.nn.regularization import createRegularizer
from ml_qm.pt.nn.ani_net import ANINetInitializer
from ml_qm.pt.scheduler import create_scheduler
from ml_qm.pt.nn.CheckPoint import DescCompCheckPointer
from ConfigSpace.configuration_space import ConfigSpace
from ml_qm.pt.nn.loss import createLossFunction
log = logging.getLogger(__name__)





class PyTorchWorker(Worker):
    def __init__(self, args:Dict[str,Any], **kwargs):
        super().__init__(**kwargs)

        self.fullid = kwargs['run_id']
        if 'id' in kwargs: self.fullid = f"{self.fullid}_{kwargs['id']}"

        log_helper.initialize_loggger(__name__, args.logFile)

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
        self.device = device
        devId = ""
        if str(device) == "cuda":
            devId = torch.cuda.current_device()
        log.info("nGPU=%i Device=%s %s" % (args.nGPU,device,devId))

        log.info("Reading conf from %s" % args.jsonFile)
        with open(args.jsonFile) as jFile:
            self.conf = json.load(jFile)

        self.trainDS, self.valDS, self.testDS = None, None, None
        if args.loadDataOnce:
            _, _, self.trainDS, self.valDS, self.testDS = self.create_data_loaders(self.conf)


    def combine_config(self, config:ConfigSpace) -> Dict[str,Any]:
        conf = copy.deepcopy(self.conf)
        for k in config.keys():
            kNames = k.split(".")
            parent = conf
            for kn in kNames[0:len(kNames)-1]:
                parent = parent[kn]
            parent[kNames[-1]] = config[k]

        return conf


    def create_data_loaders(self, conf:Dict[str,Any]):
        initAni     = ANINetInitializer(conf,self.device)
        descComput  = initAni.create_descriptor_computer()
        trainDS, valDS, testDS = createDataLoaders(descComput, conf, self.device)

        return initAni, descComput, trainDS, valDS, testDS


    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST data set.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """

        conf = self.combine_config(config)

        if not self.trainDS:
            initAni, descComput, trainDS, valDS, testDS = self.create_data_loaders(conf)
            initAni.conf = conf
        else:
            initAni    = ANINetInitializer(conf,self.device)
            descComput = initAni.create_descriptor_computer()
            trainDS, valDS, testDS = self.trainDS, self.valDS, self.testDS

        lossF       = createLossFunction(conf)
        regularizer = createRegularizer(conf)
        model       = initAni.create_model(None, descComput, False)
        optimizer   = createOptimizer(model, descComput, conf)
        scheduler   = create_scheduler(optimizer, conf)


        model = model.to(self.device)
        descComput.to(self.device)

        cid = kwargs['config_id']
        checkpoint_dir = f"checkpoints_{self.fullid}.{cid[0]}.{cid[1]}.{cid[2]}"
        i=0
        while os.path.exists(f"{checkpoint_dir}.{i}"):
            i += 1
        checkpoint_dir = f"{checkpoint_dir}.{i}"

        maxEpoch = int(budget)
        min_save_val = conf.get('minValSave', 0)
        log.info("Starting Training (training batches=%i, val batches=%i)" % ( len(trainDS), len(valDS)))
        checkPointer = DescCompCheckPointer(model, descComput, optimizer, scheduler,
                                            self.device, min_save_val, checkpoint_dir=checkpoint_dir)
        restored = checkPointer.restoreLast()

        dLoaders = {'train': trainDS, 'val':   valDS }

        epoch, model = train_model(model, descComput, checkPointer, self.device, dLoaders, lossF,
                        optimizer, scheduler, regularizer, num_epochs=maxEpoch)

        res = {
            'loss': checkPointer.best_smooth_val_mse, # remember: HpBandSter always minimizes!
            'info': {   'epochs': epoch,
                        'smoothed_val_mse': checkPointer.best_smooth_val_mse,
                        'val_mse': checkPointer.bestValMSE,
                        'max_mse': checkPointer.bestMaxMSE,
                        'train_mse': checkPointer.bestTrainMSE,
                        'number of parameters': model.number_of_parameters()
                        }
        }
        print(f"Completed: {self.fullid}: {res}")
        return res




if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="")

    parser.add_argument('-json' ,  metavar='filename' ,  type=str, dest='jsonFile', required=True,
                        help='input file')

    parser.add_argument('-nCPU' ,  metavar='n' ,  type=int, default=0,
                        help='number of CPUs')

    parser.add_argument('-nGPU' ,  metavar='n' ,  type=int, default=0,
                        help='number of GPUs, (currently only one supported)')

    parser.add_argument('-logINI',  metavar='fileName',dest='logFile', type=str,
                        help='ini file for logging (existing file or one of silent,debug)')

    args = parser.parse_args()


    worker = PyTorchWorker(args, run_id='0')
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=2, working_directory='.')
    print(res)
