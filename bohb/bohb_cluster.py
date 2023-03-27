#!/bin/env python
"""
Example 4 - on the cluster
==========================

This example shows how to run HpBandster in a cluster environment.
The actual python code does differ substantially from example 3, except for a
shared directory that is used to communicate the location of the nameserver to
every worker, and the fact that the communication is done over the network instead
of just the loop back interface.


To actually run it as a batch job, usually a shell script is required.
Those differer slightly from scheduler to scheduler.
Here we provide an example script for the Sun Grid Engine (SGE), but adapting that to
any other scheduler should be easy.
The script simply specifies the logging files for output (`-o`) and error `-e`),
loads a virtual environment, and then executes the master for the first array task
and a worker otherwise.
Array jobs execute the same source multiple times and are bundled together into one job,
where each task gets a unique task ID.
For SGE those IDs are positive integers and we simply say the first task is the master.


.. code-block:: bash

   # submit via qsub -t 1-4 -q test_core.q example_4_cluster_submit_me.sh


   # enter the virtual environment
   source ~sfalkner/virtualenvs/HpBandSter_tests/bin/activate


   if [ $SGE_TASK_ID -eq 1]
      then python3 example_4_cluster.py --run_id $JOB_ID --nic_name eth0 --working_dir .
   else
      python3 example_4_cluster.py --run_id $JOB_ID --nic_name eth0  --working_dir . --worker
   fi

You can simply copy the above code into a file, say submit_me.sh, and tell SGE to run it via:

.. code-block:: bash

   qsub -t 1-4 -q your_queue_name submit_me.sh


Now to the actual python source:
"""


import logging
import os
from bohb.pytorch_worker import PyTorchWorker

import argparse
import pickle
import time
import importlib.util

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB
from cdd_chem.util import log_helper
from py._builtin import execfile

log = logging.getLogger(__name__)



parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=9)
parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=243)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=4)
parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=2)
parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
parser.add_argument('--run_id',  type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
parser.add_argument('--host', type=str, help='hostname', required=True)
parser.add_argument('--shared_directory',type=str, help='A directory that is accessible for all processes, e.g. a NFS share.')


# trainMEM args
parser.add_argument('-confSpacePy' ,  metavar='filename' ,  type=str, required=True,
                    help='py file containing static get_configspace() function returnign configspace for optimization')

parser.add_argument('-json' ,  metavar='filename' ,  type=str, dest='jsonFile', required=True,
                    help='training conf file')
parser.add_argument('-nCPU' ,  metavar='n' ,  type=int, default=0,
                    help='number of CPUs')
parser.add_argument('-nGPU' ,  metavar='n' ,  type=int, default=0,
                    help='number of GPUs, (currently only one supported)')
parser.add_argument('-loadDataOnce',  action='store_true', default=False,
                        help='Load data once for each worker, dataloader must not depend on ConfigurationSpace')
parser.add_argument('-logINI',  metavar='fileName',dest='logFile', type=str,
                        help='ini file for logging (existing file or one of silent,debug)')
args=parser.parse_args()
log_helper.initialize_loggger(__name__, args.logFile)


if args.worker:
    time.sleep(5)	# short artificial delay to make sure the nameserver is already running
    w = PyTorchWorker(args, run_id=args.run_id, host=args.host )
    w.load_nameserver_credentials(working_directory=args.shared_directory)
    w.run(background=False)
    exit(0)


spec = importlib.util.spec_from_file_location("csg", args.confSpacePy)
csg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(csg)

config_space = csg.get_configspace()


# Start a nameserver:
# We now start the nameserver with the host name from above and a random open port (by setting the port to 0)
NS = hpns.NameServer(run_id=args.run_id, host=args.host, port=0, working_directory=args.shared_directory)
ns_host, ns_port = NS.start()

# Most optimizers are so computationally inexpensive that we can affort to run a
# worker in parallel to it. Note that this one has to run in the background to
# not block!
w = PyTorchWorker(args, run_id=args.run_id, host=args.host,
                  nameserver=ns_host, nameserver_port=ns_port)
w.run(background=True)

result_logger = hpres.json_result_logger(directory=args.shared_directory, overwrite=False)

# Run an optimizer
# We now have to specify the host, and the nameserver information
bohb = BOHB(  configspace = config_space,
                          run_id = args.run_id,
                          host=args.host,
                          nameserver=ns_host,
                          nameserver_port=ns_port,
                          min_budget=args.min_budget, max_budget=args.max_budget,
              result_logger=result_logger
              )
res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)


# In a cluster environment, you usually want to store the results for later analysis.
# One option is to simply pickle the Result object
with open(os.path.join(args.shared_directory, 'results.pkl'), 'wb') as fh:
    pickle.dump(res, fh)


# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
bohb.shutdown(shutdown_workers=True)
NS.shutdown()
