import sys
sys.path.append("..")

import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from src.utils import tensor2cuda


class ACE_PGD():
    def __init__(self, model, steps, epsilon, alpha, max_iters, _type='linf'):
        self.model = model
        # self.model.eval()

        # steps
        self.steps = steps
        
        # Maximum perturbation
        self.epsilon = epsilon
        # Movement multiplier per iteration
        self.alpha = alpha
        # Maximum numbers of iteration to generated adversaries
        self.max_iters = max_iters

    
    def CF(self, img, param):
        param=param[:,:,None,None]
        color_curve_sum = torch.sum(param, 4) + 1e-30
        total_image = img * 0
        for i in range(self.steps):
            total_image += torch.clamp(img - 1.0 * i /self.steps, 0, 1.0 / self.steps) * param[:, :, :, :, i]
        total_image *= self.steps/ color_curve_sum 
        return total_image    
    def perturb(self, original_images, labels):

        self.model.eval()

#         with torch.enable_grad():
        labels_onehot = torch.zeros(labels.size(0), 10).cuda()
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))
        Paras=torch.ones(original_images.size(0),3,self.steps).cuda()*1/self.steps
        Paras.requires_grad=True            

        prev = float('inf')
        best_adversary = original_images.clone()
        best_adversary.requires_grad = True 
        flag=torch.zeros(labels.size(0),dtype=torch.bool).cuda()
        batch_size=original_images.size(0)
        loc_suc=np.ones(batch_size)*float('inf')# the minimal number of iterations for each image to success 
        for iteration in range(self.max_iters):
            X_adv = self.CF(original_images, Paras)  

            logits = self.model(X_adv)
            real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            other = (logits - labels_infhot).max(1)[0]
            loss = torch.clamp(real - other, min=0).sum()

            loss.backward()
            grad_a=Paras.grad.clone()
            Paras.data=Paras.data-self.alpha * (grad_a.permute(1,2,0)/(torch.norm(grad_a.view(original_images.size(0),-1),dim=1)+0.00000001)).permute(2,0,1)
            Paras.grad.zero_()
            Paras.data=torch.clamp(Paras.data,min=1/self.steps,max=1/self.steps*self.epsilon)

            if iteration % 25 == 0:
                if loss > 0.9999*prev:
                    break
                prev = loss
#                 predicted_classes = (model(X_adv)).argmax(1)
            is_adv = ((self.model(X_adv)).argmax(1) != labels)
            flag=flag + is_adv
            loc_suc[is_adv.cpu()]=np.minimum((np.ones(batch_size)*iteration)[is_adv.cpu()],loc_suc[is_adv.cpu()])

            best_adversary.data[is_adv] = X_adv.data[is_adv]
        best_adversary.data[~flag]=X_adv.data[~flag]
#         iter_suc_batch=np.histogram(loc_suc,range=(0,self.max_iters),bins=10)[0]
        self.model.train()

        return best_adversary,loc_suc
