from torch.optim import Optimizer
import math
import torch


class CosineLRWithRestarts():
    """Decays learning rate with cosine annealing, normalizes weight decay
    hyperparameter value, implements restarts.
    https://arxiv.org/abs/1711.05101

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        batch_size: training samples per minibatch
        epoch_size: training samples per epoch
        restart_period: epoch count in the first restart period
        t_mult: multiplication factor by which the next restart period will extend/shrink
        eta_threshold: when this epoch is reached the min and max lr is reduced strongly every restart
        eta_min=0. initial minmum eta

    Example:
        >>> scheduler = CosineLRWithRestarts(optimizer, 32, 1024, restart_period=5, t_mult=1.2, eta_min=0.)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>         ...
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.batch_step()
        >>>     validate(...)
    """

    def __init__(self, optimizer, batch_size, epoch_size, restart_period=100,
                 t_mult=2, last_epoch=-1, eta_threshold=1000, eta_min=0., verbose=False):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an"
                                   " optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'],
                                 optimizer.param_groups))

        self.last_epoch = last_epoch
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.eta_threshold = eta_threshold
        self.eta_min = eta_min
        self.t_mult = t_mult
        self.verbose = verbose
        self.base_weight_decays = list(map(lambda group: group['weight_decay'],
                                           optimizer.param_groups))
        self.restart_period = restart_period
        self.restarts = 0
        self.t_epoch = -1

        self.step() # pytorch 1.3 changed behavior of schedulers this is workaround

    def state_dict(self):
        """Returns the state of the schedler as a :class:`dict`.
        """
        return {
            'last_epoch'     : self.last_epoch,
            'restart_period' : self.restart_period,
            'restarts'       : self.restarts,
            't_epoch'        : self.t_epoch,
        }

    def load_state_dict(self, state_dict):
        """Loads the CosineLRWithRestarts state.

        Arguments:
            state_dict (dict): CosineLRWithRestarts state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.last_epoch     = state_dict['last_epoch'    ]
        self.restart_period = state_dict['restart_period']
        self.restarts       = state_dict['restarts'      ]
        self.t_epoch        = state_dict['t_epoch'       ]



    def _schedule_eta(self):
        """
        Threshold value could be adjusted to shrink eta_min and eta_max values.
        """
        eta_min = self.eta_min
        eta_max = 1
        if self.restarts <= self.eta_threshold:
            return eta_min, eta_max

        d = self.restarts - self.eta_threshold
        k = d * 0.09
        return (eta_min + k, eta_max - k)


    def get_lr(self, t_cur):
        eta_min, eta_max = self._schedule_eta()

        eta_t = (eta_min + 0.5 * (eta_max - eta_min)
                 * (1. + math.cos(math.pi *
                                  (t_cur / self.restart_period))))

        weight_decay_norm_multi = math.sqrt(self.batch_size /
                                            (self.epoch_size *
                                             self.restart_period))
        lrs = [base_lr * eta_t for base_lr in self.base_lrs]
        weight_decays = [base_weight_decay * eta_t * weight_decay_norm_multi
                         for base_weight_decay in self.base_weight_decays]

        if self.t_epoch % self.restart_period < self.t_epoch:
            if self.verbose:
                print("Restart at epoch {}".format(self.last_epoch))
            self.restart_period *= self.t_mult
            self.restarts += 1
            self.t_epoch = 0

        return zip(lrs, weight_decays)

    def _set_batch_size(self):
        #pylint: disable=W0201
        d, r = divmod(self.epoch_size, self.batch_size)
        batches_in_epoch = d + 2 if r > 0 else d + 1
        self.batch_increment = (i for i in torch.linspace(0, 1,
                                                          batches_in_epoch))

    def step(self):
        self.last_epoch += 1
        self.t_epoch += 1
        self._set_batch_size()
        self.batch_step()

    def batch_step(self):
        t_cur = self.t_epoch + next(self.batch_increment)
        for param_group, (lr, weight_decay) in zip(self.optimizer.param_groups,
                                                   self.get_lr(t_cur)):
            param_group['lr'] = lr
            param_group['weight_decay'] = weight_decay
