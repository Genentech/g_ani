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
from cdd_chem.util import log_helper
import time
import logging
from typing import Dict, Any
import os
import copy
import argparse
import yaml
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import ConfigSpace

from ml_qm.pt.scheduler import create_scheduler
from ml_qm.pt.nn.CheckPoint import CheckPointer
from ml_qm.pt.nn.loss import createLossFunction
from scripts.trainDistNet import createOptimizer, createModel, train_model,\
    load_data_set
import re
import importlib
log = logging.getLogger(__name__)





class PyTorchWorker(Worker):
    def __init__(self, args:argparse.Namespace, update_config, loss_fct, **kwargs):
        super().__init__(**kwargs)

        self.fullid = kwargs['run_id']
        if 'id' in kwargs: self.fullid = f"{self.fullid}_{kwargs['id']}"

        self.update_config = update_config
        self.loss_fct = loss_fct

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

        log.info("Reading conf from %s" % args.yml)
        with open(args.yml) as yFile:
            self.conf = yaml.safe_load(yFile)

        self.batch_data_set = load_data_set(self.conf)

    def combine_config(self, config:ConfigSpace) -> Dict[str,Any]:
        conf = copy.deepcopy(self.conf)
        for k in config.keys():
            kNames = k.split(".")
            parent = conf
            for kn in kNames[0:len(kNames)-1]:
                if '[' in kn:
                    kn, idx = re.search("(.*)\\[(\\d+)]",kn).groups() #type: ignore
                    parent = parent[kn][int(idx)]
                else:
                    parent = parent[kn]
            kn = kNames[-1]
            idx = None
            try:
                if '[' in kn:
                    kn, idx = re.search("(.*)\\[(\\d+)]",kn).groups() #type: ignore
                    parent[kn][int(idx)] = config[k]
                else:
                    parent[kn] = config[k]
            except TypeError:
                log.warning(f"type error for {kn}")
                log.warning(f"  Parent: {parent} idx: {idx} k: {k}")
                log.warning(f"  config[k]: {config[k]}")
                log.warning(f"  Parent[kn]: {parent[kn]}")


        if self.update_config: self.update_config(conf)
        return conf


    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST data set.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """

        conf = self.combine_config(config)

        lossF = createLossFunction(conf)

        model, data_set, dataLoaderMap = createModel(conf, self.batch_data_set, self.device)

        optimizer = createOptimizer(model, conf)
        scheduler = create_scheduler(optimizer, conf)
        gradClipping = conf.get('gradientClippingMaxNorm', None)


        cid = kwargs['config_id']
        checkpoint_dir = f"checkpoints_{self.fullid}.{cid[0]}.{cid[1]}.{cid[2]}"
        i=0
        while os.path.exists(f"{checkpoint_dir}.{i}"):
            i += 1
        checkpoint_dir = f"{checkpoint_dir}.{i}"

        min_save_val = conf.get('minValSave', 0)
        log.info(f"Starting Training: budget={budget}" )
        checkPointer = CheckPointer(model, optimizer, scheduler,
                                            self.device, min_save_val, checkpoint_dir=checkpoint_dir)
        checkPointer.restoreLast()

        train_start = time.time()

        epoch, model = train_model(data_set, model, checkPointer, self.device, dataLoaderMap, lossF,
                        optimizer, scheduler, gradClipping=gradClipping, num_epochs=conf['epochs'],
                        bohb_batch_fraction=budget)

        train_elapsed_normalized = (time.time() - train_start)/budget

        res = {
            'loss': checkPointer.best_smooth_Loss, # remember: HpBandSter always minimizes!
            'info': {   'epochs': epoch,
                        'smoothed_val_mse': checkPointer.best_smooth_Loss,
                        'val_mse': checkPointer.bestValLoss,
                        'max_mse': checkPointer.bestMaxLoss,
                        'train_mse': checkPointer.bestTrainLoss,
                        'number of parameters': model.number_of_parameters(),
                        'normalised_time_s': train_elapsed_normalized
                        }
        }

        if self.loss_fct:
            res = self.loss_fct(res,conf)  # modify loss using by param range

        print(f"Completed: {self.fullid}: {res}")
        return res

    @staticmethod
    def get_example_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()

        lr = CSH.UniformFloatHyperparameter('optParam.lr', lower=1e-6, upper=1e-1, default_value='0.00002', log=True)
        weight_decay = CSH.UniformFloatHyperparameter('weight_decay', lower=1e-6, upper=0.1, default_value=1e-3, log=True)

        # For demonstration purposes, we add different optimizers as categorical hyperparameters.
        # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
        # SGD has a different parameter 'momentum'.
        optimizer = CSH.CategoricalHyperparameter('optType', ['AdamW', 'Ranger'])

        rgr_K = CSH.UniformIntegerHyperparameter('optParam.k', lower=4, upper=9, default_value=7, log=False)

        cs.add_hyperparameters([lr, optimizer, weight_decay, rgr_K])

        # The hyperparameter rgr_K will be used,if the configuration
        # contains 'Ranger' as optimizer.
        cond = CS.EqualsCondition(rgr_K, optimizer, 'Ranger')
        cs.add_condition(cond)

        return cs



if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="")

    parser.add_argument('-yaml' ,  metavar='filename' ,  type=str, dest='yml', required=True,
                        help='input file')

    parser.add_argument('-nCPU' ,  metavar='n' ,  type=int, default=0,
                        help='number of CPUs')

    parser.add_argument('-nGPU' ,  metavar='n' ,  type=int, default=0,
                        help='number of GPUs, (currently only one supported)')

    parser.add_argument('-logINI',  metavar='fileName',dest='logFile', type=str,
                        help='ini file for logging (existing file or one of silent,debug)')

    args = parser.parse_args()

    ## this is just for testing


    spec = importlib.util.spec_from_file_location("csg", args.confSpacePy)
    assert spec is not None
    csg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(csg) # type:ignore

    config_space = csg.get_configspace() #type: ignore
    loss_fct = None
    if "compute_loss" in csg.__dict__:
        loss_fct = csg.compute_loss # type:ignore
    update_config = None
    if "update_config" in csg.__dict__:
        update_config = csg.update_config # type:ignore

    # worker = PyTorchWorker(args, update_config, loss_fct, run_id='0')
    # cs = worker.get_configspace()
    #
    #
    #
    #
    #
    # worker = PyTorchWorker(args, run_id='0')
    # cs = worker.get_example_configspace()
    #
    # config = cs.sample_configuration().get_dictionary()
    # print(config)
    # res = worker.compute(config=config, budget=.4, working_directory='.', config_id='test')
    # print(res)
