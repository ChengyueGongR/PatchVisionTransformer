# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
import torch.nn as nn
from typing import Iterable, Optional

import torch
import random
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import torch.nn.functional as F
from losses import DistillationLoss
import utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None):
    # TODO fix this for finetuning
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        tau = .5
        with torch.cuda.amp.autocast():
            outputs, cont_outputs = model(samples)

            # fix me: model-level contra loss
            # '''
            # pos -> the same patch in layer ith and jth
            # neg -> different patch in layer ith and jth
            n_outputs = cont_outputs[0][:, 1:, :] # nn.functional.normalize(cont_outputs[0][:, 1:, :], dim=-1)
            n_latent = cont_outputs[1][:, 1:, :] # nn.functional.normalize(cont_outputs[1][:, 1:, :], dim=-1)# .detach()
            batch_size, num_patch, num_dim = n_outputs.shape[0], n_outputs.shape[1], n_outputs.shape[2]
            '''
            avg_outputs = nn.functional.adaptive_avg_pool2d(n_outputs.transpose(1, 2).reshape(batch_size, num_dim, int(num_patch**.5), int(num_patch**.5)), (1, 1)).reshape(batch_size, num_dim) 
            avg_latent = nn.functional.adaptive_avg_pool2d(n_latent.transpose(1, 2).reshape(batch_size, num_dim, int(num_patch**.5), int(num_patch**.5)), (1, 1)).reshape(batch_size, num_dim)
            avg_outputs = nn.functional.normalize(avg_outputs, dim=-1)
            avg_latent =  nn.functional.normalize(avg_latent, dim=-1)
            pos_pair = (avg_outputs * avg_latent).sum(-1).reshape(batch_size, 1)
            neg_pair = torch.matmul(avg_outputs, avg_outputs.permute(1, 0))
            neg_pair[torch.arange(num_patch).cuda().long(), torch.arange(num_patch).cuda().long()] *= 0. 
            neg_pair[torch.arange(num_patch).cuda().long(), torch.arange(num_patch).cuda().long()] -= 1e10
            pairs = torch.cat([pos_pair, neg_pair], dim=-1).reshape(-1, batch_size+1)
            '''
            
            
            '''
            # avg_outputs = nn.functional.avg_pool2d(n_outputs.transpose(1, 2).reshape(batch_size, num_dim, int(num_patch**.5), int(num_patch**.5)), kernel_size=3, padding=1, stride=1).reshape(batch_size, num_dim, -1).transpose(1, 2)
            pos_pair = (n_outputs * n_latent).sum(-1).reshape(batch_size, num_patch, -1)
            neg_pair = torch.bmm(n_latent, n_latent.permute(0, 2, 1))
            neg_pair[:, torch.arange(num_patch).cuda().long(), torch.arange(num_patch).cuda().long()] *= 0. 
            neg_pair[:, torch.arange(num_patch).cuda().long(), torch.arange(num_patch).cuda().long()] -= 1e10

            random_latent = n_latent.reshape(-1, num_dim)[torch.randperm(batch_size * num_patch)].reshape(batch_size, num_patch, num_dim)
            inbatch_neg_pair = torch.bmm(n_latent, random_latent.permute(0, 2, 1)) 
            # neg_pair = torch.bmm(n_outputs, n_latent.permute(0, 2, 1))
            # pos_pair = (n_outputs.detach() * n_latent).sum(-1).reshape(batch_size, num_patch, 1)
            # neg_pair = torch.bmm(n_latent, n_latent.permute(0, 2, 1))

            pairs = torch.cat([pos_pair, neg_pair, inbatch_neg_pair], dim=-1).reshape(-1, num_patch*2+1)
            pusdo_label = torch.zeros_like(pairs[:, 0]).reshape(-1).long()
            '''
           
            n_outputs = nn.functional.normalize(n_outputs, dim=-1)
            n_outputs = n_outputs.transpose(1, 2).reshape(batch_size, num_dim, int(num_patch**.5), int(num_patch**.5))
            n_latent = n_latent.transpose(1, 2).reshape(batch_size, num_dim, int(num_patch**.5), int(num_patch**.5))
            
            width = int(num_patch**.5)
            center = n_outputs[:, :, 1:width-1, 1:width-1].reshape(batch_size, num_dim, -1).permute(0, 2, 1)
            # center = n_latent[:, :, 1:width-1, 1:width-1].reshape(batch_size, num_dim, -1).permute(0, 2, 1)
            pos_lst = []
            select_lst = [(0, 2), (1, 1), (2, 0)]
            for first_ind in range(3):
                for second_ind in range(3):
                    if first_ind == 1 and second_ind == 1:
                        continue
                    pos_lst.append(n_outputs[:, :, select_lst[first_ind][0]:width-select_lst[first_ind][1], 
                        select_lst[second_ind][0]:width-select_lst[second_ind][1]].reshape(batch_size, num_dim, -1).permute(0, 2, 1))
            pos_pair = torch.log(1e-10 + sum([torch.exp((center * item).sum(-1)) for item in pos_lst]) / 8).reshape(batch_size, (width - 2)**2, 1)
            n_outputs = center.reshape(batch_size, num_dim, -1).transpose(1, 2)
            

            neg_pair = torch.bmm(n_outputs, n_outputs.permute(0, 2, 1).detach())
            random_outputs = n_outputs.reshape(-1, num_dim)[torch.randperm(batch_size * (width - 2)**2)].reshape(batch_size, (width - 2)**2, num_dim)
            # random_outputs =  n_latent[:, :, 1:width-1, 1:width-1].reshape(batch_size, num_dim, -1).permute(0, 2, 1)
            inbatch_neg_pair = torch.bmm(n_outputs, random_outputs.permute(0, 2, 1).detach()) 
            
            pairs = torch.cat([pos_pair, neg_pair, inbatch_neg_pair], dim=-1).reshape(-1, (width - 2)**2*2+1)
            pusdo_label = torch.zeros_like(pairs[:, 0]).reshape(-1).long()


            '''
            # pos -> different patch in final layer for a same image
            # neg -> patchs in another image
            n_outputs = cont_outputs[0][:, 1:, :] # nn.functional.normalize(cont_outputs[0][:, 1:, :], dim=-1)
            batch_size, num_patch, num_dim = n_outputs.shape[0], n_outputs.shape[1], n_outputs.shape[2]

            # n_latent = nn.functional.normalize(cont_outputs[1][:, 1:, :], dim=-1)
            
            # num_instance = 5
            pos_outputs = nn.functional.avg_pool2d(n_outputs.transpose(1, 2).reshape(batch_size, num_dim, int(num_patch**.5), int(num_patch**.5)), kernel_size=3, padding=1, stride=1).reshape(batch_size, num_dim, -1).transpose(1, 2)
            n_outputs = nn.functional.normalize(n_outputs, dim=-1)
            pos_outputs = nn.functional.normalize(pos_outputs, dim=-1)
            pos_pair = (n_outputs * pos_outputs).sum(-1).reshape(batch_size, num_patch, -1)
            # pos_pair = (n_outputs * n_outputs[:, torch.randperm(num_patch), :]).sum(-1).reshape(batch_size, num_patch, -1)
            # pos_pair = sum([(n_outputs * n_outputs[:, torch.randperm(num_patch), :]).sum(-1).reshape(batch_size, num_patch, -1) for _ in range(num_instance)]) / num_instance
            random_outputs = n_outputs.reshape(-1, num_dim)[torch.randperm(batch_size * num_patch)].reshape(batch_size, num_patch, num_dim)
            neg_pair = torch.bmm(n_outputs, random_outputs.permute(0, 2, 1).detach())
            pairs = torch.cat([pos_pair, neg_pair], dim=-1).reshape(-1, num_patch+1)
            pusdo_label = torch.zeros_like(pairs[:, 0]).reshape(-1).long()
            '''
            
            loss = criterion(samples, outputs, targets) 
            loss_value = loss.item()

            loss += F.cross_entropy(pairs, pusdo_label) * tau
        # loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 50, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
