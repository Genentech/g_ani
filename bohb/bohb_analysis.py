#!/bin/env python
import matplotlib.pyplot as plt
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
import math
from cdd_chem.util.io import warn  # noqa: F401; # pylint: disable=W0611


class BOHBAnalysis:
    """ analyze bohb run """

    def __init__(self, result):
        self.result = result

    def runs_to_tab(self):
        info_keys = ['smoothed_val_loss', 'val_loss', 'max_loss', 'train_loss', 'epochs', 'number of parameters']
        info_keys = ['smoothed_val_loss', 'val_loss', 'max_loss', 'train_loss', 'epochs', 'number of parameters']
        info_keys = ['smoothed_val_loss', 'val_loss', 'max_loss', 'train_loss', 'test_mse', 'epochs', 'number of parameters']
        info_keys = ['smoothed_val_loss', 'val_loss', 'max_loss', 'train_loss', 'test_mse', 'epochs', 'number of parameters', "layer_param_estimate"]
        info_keys = ['smoothed_max_loss', 'val_loss', 'max_loss', 'train_loss', 'epochs', 'number of parameters', "layer_param_estimate"]
        defInfo = {k: "" for k in info_keys}

        param_keys_set = set()

        runs = self.result.data
        for cid, d in runs.items():
            param_keys_set.update(d.config.keys())
        param_keys = list(param_keys_set)
        param_keys.sort()

        headr = ['iter', 'c2', 'c3', 'cid', 'loss', 'budget', 'model_based_pick']
        headr.extend(info_keys)
        headr.extend([k.replace(".","_") for k in param_keys])
        headr.append('wall_time_s')
        headr.append('normalised_time_s')
        headr.append('exception')

        print('\t'.join(headr))

        for cid, d in runs.items():
            cnfg = d.config
            results = d.results

            if results.items():
                for b, res in results.items():
                    if not res:
                        loss = ''
                        info = defInfo
                    else:
                        info = res['info']
                        loss = res['loss']

                    exception = d.exceptions.get(b, '')
                    if exception is not None:
                        exception=exception.strip().split('\n')[-1]
                        exception=exception[0:40]
                    else:
                        exception = ''

                    line = list(cid)
                    line.append(str(cid))
                    times = d.time_stamps[b]
                    wall_time_s = times['finished'] - times['started']
                    train_elapsed_normalized = wall_time_s/b
                    line.append(loss)
                    line.append(b)
                    line.append(d.config_info['model_based_pick'])
                    line.extend([info.get(k,'') for k in info_keys])
                    line.extend([cnfg.get(k,'') for k in param_keys])
                    line.append(wall_time_s)
                    line.append(train_elapsed_normalized)
                    line.append(exception)

                    print('\t'.join(str(v) for v in line))
            else:
                line = list(cid)
                line.append(str(cid))
                line.append('') #loss
                line.append('') #b
                line.append(d.config_info['model_based_pick'])
                line.extend([defInfo.get(k, '') for k in info_keys])
                line.extend([cnfg.get(k, '') for k in param_keys])
                line.append('') # wall_time_s
                line.append('') # train_elapsed_normalized
                line.append('') #exception

                print('\t'.join(str(v) for v in line))

# load the example run from the log files
#result = hpres.logged_results_to_HBS_result('y:/gstore/home/albertgo/projects/dockingPoseAssessment/docking/ag-workflow/dockpaRun202111/learning/1/bohb.1/3/')
result = hpres.logged_results_to_HBS_result('y:/gstore/home/albertgo/projects/dockingPoseAssessment/docking/ag-workflow/dockpaRun202111/learning/b/bohb.5.41/1/')
#result = hpres.logged_results_to_HBS_result('//smddfiles.gene.com/smdd/CDD/projects/ml_qm/backup/ml_qm/dist3/bohb.ac1/2')

#with io.open("y:/gstore/home/albertgo/scratch/ml_qm/pka2/bohb.base/1/results.pkl", "rb") as infile:
#    result = pickle.load(infile)

ba = BOHBAnalysis(result)
ba.runs_to_tab()

# get all executed runs
all_runs = result.get_all_runs()

# get the 'dict' that translates config ids to the actual configurations
id2conf = result.get_id2config_mapping()


# Here is how you get he incumbent (best configuration)
inc_id = result.get_incumbent_id()

# let's grab the run on the highest budget
inc_runs = result.get_runs_by_id(inc_id)
inc_run = inc_runs[-1]


# We have access to all information: the config, the loss observed during
#optimization, and all the additional information
inc_loss = inc_run.loss
inc_config = id2conf[inc_id]['config']
inc_test_loss = inc_run.info['smoothed_max_loss']

print(f'Best found configuration, loss={inc_test_loss}:')
print(inc_config)
print('It results.'%(inc_run.info))


# Let's plot the observed losses grouped by budget,
hpvis.losses_over_time(all_runs, get_loss_from_run_fn = lambda r: math.log10(r.loss))

# the number of concurent runs,
hpvis.concurrent_runs_over_time(all_runs)

# and the number of finished runs.
hpvis.finished_runs_over_time(all_runs)

# This one visualizes the spearman rank correlation coefficients of the losses
# between different budgets.
hpvis.correlation_across_budgets(result)

# For model based optimizers, one might wonder how much the model actually helped.
# The next plot compares the performance of configs picked by the model vs. random ones
#hpvis.performance_histogram_model_vs_random(all_runs, id2conf)
