import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Optional
import torch.autograd as autograd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import os
import argparse
import AdvCF

parser = argparse.ArgumentParser(description = "main")
parser.add_argument("-batch_size", "--batch_size_ori", type=int, help="Number of images processed at one time", default=25)
parser.add_argument("-gpu", "--gpu", type=int, help="1 for gpu and -1 for cpu", default=1)
parser.add_argument("-max_iterations", "--max_iterations", type=int, default=500)
parser.add_argument("-learning_rate", "--lr", type=float, help="learning rate in Adam optimization", default=0.01)
parser.add_argument("-pieces", "--pieces", type=int, help="the number of pieces (1/s) in the piecewise-linear AdvCF", default=64)
parser.add_argument("-search_steps", "--search_steps", type=int, help="the number of steps set for searching an optimal lambda", default=1)
parser.add_argument("-initial_lambda", "--initial_lambda", type=float, default=5)

args = parser.parse_args()
batch_size=args.batch_size_ori
max_iterations=args.max_iterations
lr=args.lr
search_steps=args.search_steps
pieces=args.pieces
initial_lambda=args.initial_lambda

#initialization
trn = transforms.Compose([
     transforms.ToTensor(),])

image_id_list,label_ori_list,label_tar_list=load_ground_truth(os.path.join('./', 'images.csv'))
model = models.inception_v3(pretrained=True,transform_input=False).eval()
for param in model.parameters():
    param.requires_grad=False
model.to(device)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

output_path='./AdvCF_CW_piece_'+str(pieces)+'_lr_'+str(lr)+'_iter_'+str(max_iterations)+'_initial_lambda_'+str(initial_lambda)+'/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
num_batches = np.int(np.ceil(len(image_id_list)/batch_size))
for k in tqdm_notebook(range(0,1)):
    batch_size_cur=min(batch_size,len(image_id_list)-k*batch_size)    
    X_ori = torch.zeros(batch_size_cur,3,299,299).to(device)
    for i in range(batch_size_cur):  
        X_ori[i]=trn(Image.open('./images/'+image_id_list[k*batch_size+i]+'.png')).unsqueeze(0)  
    labels=torch.argmax(model((X_ori-0.5)/0.5),dim=1)
    #genrate the adversarial images
    approach = AdvCF(device=device,max_iterations=max_iterations,learning_rate=lr,search_steps=search_steps,initial_const=initial_lambda)
    X_adv,o_best_l2= approach.adversary(model, X_ori, labels=labels, pieces=pieces, targeted=False)

    #save the successfully adversarial images
    for j in range(batch_size_cur):
        if o_best_l2[j]<1e10:
            x_np=transforms.ToPILImage()(X_adv[j].detach().cpu())
            x_np.save(os.path.join(output_path,image_id_list[k*batch_size+j])+'.png')


  
