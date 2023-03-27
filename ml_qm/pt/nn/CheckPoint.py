"""
Helps checkpointing Pytorch NN training

Created on Jun 20, 2018

@author: albertgo
"""
import math

import time
import os
import torch
import logging
from builtins import getattr

from cdd_chem.util.io import warn # noqa: F401; # pylint: disable=W0611
from typing import List

log = logging.getLogger(__name__)

INF = float("inf")


class CheckPointer:
    """ Manage Checkpinting during model training """
    def __init__(self, model, optimizer, scheduler, device,
                 min_loss_to_save:float = 3.5, best_save_tolerance=1., intervalS=3600,
                 checkpoint_dir:str = "checkpoints", lossType="MSE"):
        """
        :param model: to be checkpointed
        :param optimizer: optimizer used
        :param scheduler: scheduler used
        :param device:    device on which model is trained
        :param min_loss_to_save: save model if loss below this threshold
        :param best_save_tolerance: save best file if loss <= min_loss_to_save*min_loss_to_save
        :param intervalS: interval [s] for saving checkpint files
        :param checkpoint_dir: directory to save too
        :param lossType: string for loss type lable
        """

        assert best_save_tolerance >= 1.

        #intervalS  = 0############################################################################
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.intervalS = intervalS
        self.lastCheckpointS = time.time()
        self.lossType = lossType
        self.max_loss_history:List[float] = []
        self.min_loss_to_save = min_loss_to_save
        self.best_save_tolerance = best_save_tolerance


        self.bestTrainLoss = INF

        self.bestValLoss = INF
        self.bestMaxLoss = INF

        self.smoothing_window = 3
        self.best_smooth_Loss = INF

        self.epochStart = time.time()
        self.numEpoch = 0

        self.directory = checkpoint_dir
        os.makedirs(self.directory, exist_ok=True)


    def checkPoint(self, trainLoss, valLoss=float("inf")):
        """ Save checkpoint if intervalS seconds passed and increment epoch counter

            This should be called for every epoch.
            Writing of checkpoints is done only if intervalS sec have past
        """

        best_smooth = False
        if valLoss < INF:
            maxLoss = max(valLoss,trainLoss)
            if maxLoss < self.bestMaxLoss: self.bestMaxLoss = maxLoss

            if (not math.isnan(maxLoss)) or len(self.max_loss_history) == 0:
                self.max_loss_history.append(maxLoss)

            smooth_elements = self.max_loss_history[
                max(0, len(self.max_loss_history) - self.smoothing_window):len(self.max_loss_history)]
            smooth_Loss = sum(smooth_elements)/len(smooth_elements)

            if smooth_Loss <= self.best_smooth_Loss * self.best_save_tolerance:
                if smooth_Loss < self.best_smooth_Loss:
                    self.best_smooth_Loss = smooth_Loss

                if smooth_Loss < self.min_loss_to_save: self.save_best()

                best_smooth = True

            if valLoss < self.bestValLoss: self.bestValLoss = valLoss
            if valLoss <= self.bestValLoss * self.best_save_tolerance \
               and valLoss < self.min_loss_to_save and not best_smooth:
                self.save_best()

        if trainLoss < self.bestTrainLoss: self.bestTrainLoss = trainLoss

        if time.time() - self.lastCheckpointS < self.intervalS:
            return

        # checkpoint full current state
        self.saveCheckPoint()


    def printResults(self, trnLoss, valLoss, tstLoss):
        t = time.time()
        epochTime = t - self.epochStart
        lr = self._getLR()

        sMaxLoss = INF
        if self.max_loss_history:
            smooth_elements = self.max_loss_history[
                max(0, len(self.max_loss_history) - self.smoothing_window):len(self.max_loss_history)]
            sMaxLoss = sum(smooth_elements)/len(smooth_elements)

        valLoss  = "%-7.3f" % valLoss  if valLoss  is not None else "       "
        sMaxLoss = "%-7.3f" % sMaxLoss if sMaxLoss is not INF  else "       "
        tstLoss  = "%-7.3f" % tstLoss  if tstLoss  is not None else "       "
        bMaxLoss = "%-7.3f" % self.bestMaxLoss if self.bestMaxLoss is not None else "       "
        trnLoss  = "%-9.3f" % trnLoss if trnLoss>0.0001 else "%-9g" % trnLoss
        print(f'Epoch %3i %5is {self.lossType} tr: %s V: %s sM: %s Tst: %s   Mx: %s  lr: %s' %
              (self.numEpoch, epochTime, trnLoss, valLoss, sMaxLoss, tstLoss, bMaxLoss, lr),
              flush=True )


    def get_additional_state(self):
        """ get any additional state to store. Overwrite if necessary """
        return {}


    def save_best(self):
        cFile = "%s/best_%06i.nnp"    % (self.directory,self.numEpoch)
        cp = {
            "epoch": self.numEpoch,
            "model": self.model.state_dict(),
            f"val{self.lossType}" : self.bestValLoss,
            f"max{self.lossType}" : self.bestMaxLoss,
        }
        cp.update(self.get_additional_state())
        torch.save(cp, "%s/besttmp_%06i" % (self.directory,self.numEpoch) )

        os.replace("%s/besttmp_%06i" % (self.directory,self.numEpoch), cFile)


    def saveCheckPoint(self):
        """ immediately save checkpoint of current model state """

        scd = None
        # ReduceLROnPlateau dose not have state_dict
        if getattr(self.scheduler, 'state_dict', False): scd = self.scheduler.state_dict()

        cp = {
            "epoch":     self.numEpoch,
            "model":     self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": scd,
            f"best{self.lossType}":   self.bestValLoss,
            f"max{self.lossType}":    self.bestMaxLoss,
            f"smooth{self.lossType}": self.best_smooth_Loss,
            f"al_{self.lossType.lower()}_history": self.max_loss_history,
        }
        cp.update(self.get_additional_state())
        torch.save(cp, "%s/chkptytmp_%06i" % (self.directory, self.numEpoch) )

        os.replace("%s/chkptytmp_%06i" % (self.directory, self.numEpoch),
                  "%s/chkpty_%06i"    % (self.directory,  self.numEpoch))

        self.lastCheckpointS = time.time()


    def incrementEpoch(self):
        self.numEpoch += 1
        self.epochStart = time.time()


    def _getLR(self): # copied from _LRScheduler
        strLR = ""
        for param_group in self.optimizer.param_groups:
            lr = float(param_group['lr'])
            strLR += "%g " % lr
        return strLR

    def printWeights(self):
        for name, weights in self.model.named_parameters():
            print("%s: %s" % (name,weights), flush=True)

    def find_checkpoint(self):
        """ look for the latest checkpoint """
        if not os.path.exists(self.directory): return None

        files = [f for f in os.listdir(self.directory) if f.startswith("chkpty_") ]
        if len(files) == 0: return None

        files.sort(reverse = True)
        last = "%s/%s" % (self.directory,files[0])

        if not os.path.isfile(last):
            log.critical("=> no checkpoint found at '%s'" % last)
            return None

        log.info("=> loading %s" % last)
        return torch.load(last, map_location='cpu')


    def restore_checkpoint(self, checkpoint):
        epoch = checkpoint['epoch']

        self.model.load_state_dict(checkpoint['model'])
        self.model = self.model.to(self.device)

        oldState = self.optimizer.state_dict()
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            log.critical("Error reading optimizer state, restoring from conf settings: %s" % str(e))
            self.optimizer.load_state_dict(oldState)

        if self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            except (AttributeError, TypeError) as e:
                log.warning("warning: %s ignored for %s: %s" % ("AttributeError", type(self.scheduler), str(e)))

        self.numEpoch = epoch
        try:
            self.bestValLoss = checkpoint[f'best{self.lossType}']
            self.bestMaxLoss = checkpoint[f'max{self.lossType}']
            self.best_smooth_Loss = checkpoint[f"smooth{self.lossType}"]
            self.max_loss_history     = checkpoint[f"val_{self.lossType.lower()}_history"]
        except KeyError:
            pass

        self.epochStart = time.time()


    def restoreLast(self):
        """ look for the latest checkpoint and restore its state if present """
        checkpoint = self.find_checkpoint()
        if not checkpoint: return False

        self.restore_checkpoint(checkpoint)
        return True


class DescCompCheckPointer(CheckPointer):
    """ CheckPointer that takes care of parameters in DescriptorComputer for trainMEM.py """

    def __init__(self, model, descComputer, optimizer, scheduler, device,
                 min_val_loss_to_save:float = 3.5, best_save_tolerance=1., intervalS=3600, checkpoint_dir:str = "checkpoints"):
        super().__init__(model, optimizer, scheduler, device, min_val_loss_to_save, best_save_tolerance, intervalS, checkpoint_dir)
        self.descComputer = descComputer


    def get_additional_state(self):
        return { "descComputer": self.descComputer.state_dict() }


    def printWeights(self):
        super().printWeights()
        print("-"*30)
        self.descComputer.printOptParam()
        print("="*30, flush=True)


    def restoreLast(self):
        """ look for the latest checkpoint and restore its state if present """
        checkpoint = self.find_checkpoint()
        if not checkpoint: return False

        self.restore_checkpoint(checkpoint)
        self.descComputer.load_state_dict(checkpoint['descComputer'])
        self.descComputer.to(self.device)
        return True


    def printBasisOptParam(self):
        self.descComputer.printOptParam()
