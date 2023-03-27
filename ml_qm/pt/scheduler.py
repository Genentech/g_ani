import torch.nn
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingWarmRestarts, StepLR
from typing import Dict
import logging

from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611
from ml_qm.pt.cosine_scheduler import CosineLRWithRestarts as github_CosineLRWithRestarts

log = logging.getLogger(__name__)



def create_scheduler(optimizer, conf):
    # create scheduler from conf dict
    scheduler_name = conf.get('scheduler', None)
    if scheduler_name is None: return None
    if scheduler_name == "CombinedScheduler":
        scheduler = CombinedScheduler
    else:
        scheduler = globals().get(scheduler_name, None)
        if scheduler is None:
            scheduler = getattr(lr_scheduler, scheduler_name, None)
        if scheduler is None:
            raise TypeError("unknown scheduler: " + scheduler_name)

    init_lr = None
    if scheduler is not None:
        init_lr = conf.get('init_lr', None)
        if init_lr:
            for group in optimizer.param_groups:
                del group['initial_lr']
                group['lr'] = init_lr
        scheduler = scheduler(optimizer, **conf['schedulerParam'])

    log.warning(f"New scheduler: {scheduler} lr: {init_lr}") # pylint: disable=W1203

    return scheduler


def _printLR(optimizer): # copied from _LRScheduler
    strLR = ""
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        strLR += "%g " % lr
    warn( strLR )



class CombinedScheduler(_LRScheduler):
    """ Scheduler that combines multiple schedulers
        Argument a dict of scheduler descriptions keyed by either epoch number or minimum train loss.

    Example:
        { 0: { "scheduler": "StepLR",               "schedulerParam": { step_size: 30 }},
          5: { "scheduler": "CosineLRWithRestarts", "schedulerParam": { "batch_size":2560, "epoch_size": 6000000, "restart_period": 5, "t_mult":1.2, "verbose":true }}}
    """

    def __init__(self, optimizer, conf: Dict[int,Dict],last_epoch=-1):
        self.current_scheduler = None
        self.current_sched_idx = 0
        self.optimizer = optimizer
        self.switch_epochs = []
        self.scheduler_desc = []
        self.first_epoch = None
        self.global_epoch = 0

        for e, s in conf.items():
            self.switch_epochs.append(int(e))
            self.scheduler_desc.append( s )

        super().__init__(optimizer,last_epoch)


    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer, switch_epoch, 'scheduler_desc'.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.
        """
        cur_sched = self.current_scheduler
        self.current_scheduler = None
        state = super().state_dict()
        del state['switch_epochs'], state['scheduler_desc']
        self.current_scheduler = cur_sched
        state['current_scheduler'] = self.current_scheduler.state_dict()

        return state

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        cur_sched = state_dict['current_scheduler']
        del state_dict['current_scheduler']
        super().load_state_dict(state_dict)
        self.current_scheduler = create_scheduler(
                                    self.optimizer, self.scheduler_desc[self.current_sched_idx-1] )
        self.current_scheduler.load_state_dict(cur_sched)


    def get_lr(self):
        raise NotImplementedError


    def step(self, epoch=None):
        # pylint: disable=R1710
        # did we reach the next scheduler switch
        if len(self.switch_epochs) > self.current_sched_idx \
           and self.global_epoch == self.switch_epochs[self.current_sched_idx]:

            for _, group in enumerate(self.optimizer.param_groups):
                group['initial_lr'] = group['lr']

            self.current_scheduler = create_scheduler(
                                        self.optimizer, self.scheduler_desc[self.current_sched_idx] )
            # In combinedScheduler the epoch passed might not be 0 based
            # as we might be the second or later scheduler, remember when we started
            self.first_epoch = self.global_epoch
            self.current_sched_idx += 1

            self.last_epoch = self.current_scheduler.last_epoch
            self.global_epoch += 1
            return None  # self.current_scheduler.step(epoch) has already been called

        ret = None
        if self.current_scheduler:
            ret = self.current_scheduler.step(epoch)
            self.last_epoch = self.current_scheduler.last_epoch

        self.global_epoch += 1

        return ret


    def batch_step(self, global_epoch=None):
        if getattr(self.current_scheduler,'batch_step', False):
            self.current_scheduler.batch_step(global_epoch - self.first_epoch)


class CosineWarmRestarts(CosineAnnealingWarmRestarts):
    ''' Wrapper arount pytorchs:CosineAnnealingWarmRestarts
        Our code depends on batch_step() for schedulers that are called each minibatch '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batch_step(self, epoch=None):  # pylint: disable=W0221,W0613
        super().step(epoch)

    def step(self, epoch=None):  # pylint: disable=W0221,W0613
        return


class CosineLRWithRestarts(github_CosineLRWithRestarts):
    ''' Wrapper around CosineLRWithRestarts from github
        Our code depends on batch_step() for schedulers that are called each minibatch '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batch_step(self, epoch=None):  # pylint: disable=W0221,W0613
        super().batch_step()

    def step(self, epoch=None):  # pylint: disable=W0221,W0613
        return super().step()


class BatchStepLR(StepLR):
    ''' A StepLR that is based on minibatches instead of epochs '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batch_step(self, epoch=None): # pylint: disable=W0221,W0613
        super().step()

    def step(self, epoch=None):  # pylint: disable=W0221,W0613
        return
