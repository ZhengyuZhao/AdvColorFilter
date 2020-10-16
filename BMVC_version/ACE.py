"""
The code is heavily based on https://github.com/jeromerony/fast_adversarial/blob/master/fast_adv/adversarys/carlini.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Optional
import torch.autograd as autograd
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import os
from utils import *


class ACE:
    """
    Parameters
    ----------
    num_classes : int
        Number of classes of the model to adversary.
    confidence : float, optional
        Confidence of the adversary for Carlini's loss, in term of distance between logits.
    learning_rate : float
        Learning rate for the optimization.
    search_steps : int
        Number of search steps to find the best scale constant for Carlini's loss.
    max_iterations : int
        Maximum number of iterations during a single search step.
    initial_const : float
        Initial constant of the adversary.
    device : torch.device, optional
        Device to use for the adversary.
    pieces : int
        the number of pieces in the piecewise-linear color filter
    
    """

    def __init__(self,
                 num_classes: int = 1000,
                 confidence: float = 0,
                 learning_rate: float = 0.001,
                 search_steps: int = 9,
                 max_iterations: int = 1000,
                 abort_early: bool = True,
                 initial_const: float = 10,
                 device: torch.device = torch.device('cpu')
                ) -> None:

        self.confidence = confidence
        self.learning_rate = learning_rate

        self.binary_search_steps = search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.num_classes = num_classes

        self.repeat = self.binary_search_steps >= 10


        self.device = device

    def _step(self, model: nn.Module, optimizer: optim.Optimizer, inputs: torch.Tensor, Paras: torch.Tensor, pieces: float,
               labels: torch.Tensor, labels_infhot: torch.Tensor, targeted: bool,
              const: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = inputs.shape[0]


        Paras.data=torch.clamp(Paras.data,min=0)
        Paras_sum=torch.sum(Paras.view(batch_size,3,-1),dim=2,keepdim=True)
        
        # regularization on the adjustment
        l2 = torch.sum((((Paras/Paras_sum-1/pieces)**2)).view(batch_size,-1),dim=1)

        adv = CF(inputs, Paras,pieces)
        logits = model(norm(adv))

        real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        other = (logits - labels_infhot).max(1)[0]
        if targeted:
            # if targeted, optimize for making the other class most likely
            logit_dists = torch.clamp(other - real + self.confidence, min=0)
        else:
            # if non-targeted, optimize for making this class least likely.
            logit_dists = torch.clamp(real - other + self.confidence, min=0)

        logit_loss = (logit_dists).sum()
        l2_loss = (const * l2).sum()
        loss = logit_loss + l2_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return quantization(adv).detach(), l2.detach(), loss.detach()

    def adversary(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,pieces: float,
               targeted: bool = False) ->  Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs ACE given the inputs and labels.

        Parameters
        ----------
        model : nn.Module
            Model to fool.
        inputs : torch.Tensor
            Batch of image examples.
        labels : torch.Tensor
            Original labels if untargeted, else labels of targets.
        pieces :the number of pieces in the piecewise-linear color filter    
        targeted : bool, optional
            Whether to perform a targeted adversary or not.

        Returns
        -------
        torch.Tensor
            Batch of image samples modified to be adversarial
        """
        batch_size = inputs.shape[0]

        # set the lower and upper bounds accordingly
        lower_bound = torch.zeros(batch_size, device=self.device)
        CONST = torch.full((batch_size,), self.initial_const, device=self.device)
        upper_bound = torch.full((batch_size,), 1e10, device=self.device)

        o_best_l2 = torch.full((batch_size,), 1e10, device=self.device)
        o_best_score = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)
        o_best_adversary = inputs.clone()

        # setup the target variable, we need it to be in one-hot form for the loss function
        labels_onehot = torch.zeros(labels.size(0), self.num_classes, device=self.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))

        for outer_step in range(self.binary_search_steps):

            # setup the optimizer
            Paras=torch.ones(batch_size,3,pieces).to(self.device)*1/pieces
            Paras.requires_grad=True            
            optimizer = optim.Adam([Paras], lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)
            best_l2 = torch.full((batch_size,), 1e10, device=self.device)
            best_score = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)

            # The last iteration (if we run many pieces) repeat the search once.
            if self.repeat and outer_step == (self.binary_search_steps - 1):
                CONST = upper_bound

            prev = float('inf')
            for iteration in range(self.max_iterations):
                # perform the adversary
                adv, l2, loss = self._step(model, optimizer, inputs, Paras, pieces, labels, labels_infhot, targeted, CONST)

                # check if we should abort search if we're getting nowhere.
                if self.abort_early and iteration % (self.max_iterations // 20) == 0:
                    if loss > prev * 0.9999:
                        
                        break
                    prev = loss

                # adjust the best result found so far

                predicted_classes = (model(norm(adv)) - labels_onehot * self.confidence).argmax(1) if targeted else \
                    (model(norm(adv)) + labels_onehot * self.confidence).argmax(1)

                is_adv = (predicted_classes == labels) if targeted else (predicted_classes != labels)
                    
                is_smaller = l2 < best_l2
                o_is_smaller = l2 < o_best_l2
                is_both = is_adv * is_smaller
                o_is_both = is_adv * o_is_smaller

                best_l2[is_both] = l2[is_both]
                best_score[is_both] = predicted_classes[is_both]
                o_best_l2[o_is_both] = l2[o_is_both]
                o_best_score[o_is_both] = predicted_classes[o_is_both]
                o_best_adversary[o_is_both] = adv[o_is_both]
            # adjust the constant as needed
            adv_found = (best_score == labels) if targeted else ((best_score != labels) * (best_score != -1))
            upper_bound[adv_found] = torch.min(upper_bound[adv_found], CONST[adv_found])
            adv_not_found = ~adv_found
            lower_bound[adv_not_found] = torch.max(lower_bound[adv_not_found], CONST[adv_not_found])
            is_smaller = upper_bound < 1e9
            CONST[is_smaller] = (lower_bound[is_smaller] + upper_bound[is_smaller]) / 2
            CONST[(~is_smaller) * adv_not_found] *= 10

        # return the best solution found
        return o_best_adversary,o_best_l2
