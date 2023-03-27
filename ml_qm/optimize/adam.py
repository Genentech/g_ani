import math
import torch
from ml_qm.pt.nn.ani_net import SameSizeCoordsBatch
from t_opt.opt_util import DEFAULT_CONVERGENCE_OPTS


import logging
log = logging.getLogger(__name__)

class BatchAdam():
    r"""Implements Adam algorithm. On a batch of conformations

    This code is abandond in favor of batch_lbfgs since it needs only about
    Half as many steps.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, lr:float = 0.05,
                 convergence_opts = DEFAULT_CONVERGENCE_OPTS,
                 betas=(0.9, 0.999), eps=1e-8,
                 amsgrad=False):
        #pylint: disable=C0113,C0122
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.amsgrad = amsgrad
        self.beta1, self.beta2 = betas
        self.lr = lr
        self.eps=eps
        self.convergence_opts = convergence_opts

    def optimize(self, coords_batch:SameSizeCoordsBatch, energyHelper):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        n_step = 0
        coords = coords_batch.coords
        #n_confs = coords.shape[0]
        exp_avg = torch.zeros_like(coords)
        exp_avg_sq = torch.zeros_like(coords)
        #prev_loss = torch.full((n_confs,), 1e19, dtype=coords.dtype, device=coords.device)
        if self.amsgrad:
            max_exp_avg_sq = torch.zeros_like(coords)

        while True:
            n_step += 1
            loss, _ = energyHelper.compute_energy()
            grad = energyHelper.compute_grad()

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(self.beta1).add_(1 - self.beta1, grad)
            exp_avg_sq.mul_(self.beta2).addcmul_(1 - self.beta2, grad, grad)
            if self.amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = max_exp_avg_sq.sqrt().add_(self.eps)
            else:
                denom = exp_avg_sq.sqrt().add_(self.eps)

            bias_correction1 = 1 - self.beta1 ** n_step
            bias_correction2 = 1 - self.beta2 ** n_step
            step_size = self.lr * math.sqrt(bias_correction2) / bias_correction1

            coords.data.addcdiv_(-step_size, exp_avg, denom)

            #deltaLoss = prev_loss - loss
            #deltaLossS = deltaLoss * deltaLoss
            #dummy = deltaLossS < self.convergence_opts.convergence_es
            if log.isEnabledFor(logging.DEBUG):
                # pylint: disable=W0613,W1203
                #                 log.debug(f'{n_step} loss {loss[0:5].detach().cpu().numpy()} grad {grad[0:5][0].detach().cpu().numpy()}')
                log.debug(f'{n_step} loss {loss[0:5].detach().cpu().numpy()}')

        return loss
