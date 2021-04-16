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
from loss_ops import alpha_divergence
import numpy as np
from scipy.stats import truncnorm

def generate_mask(mask, num_patch, sel_rate):
    max_length = min(num_patch, int(num_patch * num_patch * sel_rate))
    length = random.randint(1, max(1, max_length - 1))
    width = int(num_patch * num_patch * sel_rate) // length
    if width > num_patch:
        width = num_patch
        length = int(num_patch * num_patch * sel_rate) // width
    length_left_ind = random.randint(0, max(0, num_patch - length))
    length_right_ind = length_left_ind + length
    width_left_ind = random.randint(0, max(0, num_patch - width))
    width_right_ind = width_left_ind + width

    mask = torch.zeros_like(mask)
    mask[length_left_ind:length_right_ind, width_left_ind:width_right_ind] = torch.ones_like(mask[length_left_ind:length_right_ind, width_left_ind:width_right_ind])
    return mask

def multi_mix(samples, targets, num_patch=14, num_mix=8):
    mix_rate = [1.] * num_mix
    mix_rate[-1] += num_mix * 2
    mix_rate = np.array(mix_rate) / sum(mix_rate)
    # mix_rate = [1] * num_mix
    # mix_rate[-1] += int(num_mix * 1.5) # (num_mix - 1) // 2
    # mix_rate = np.random.dirichlet(mix_rate, 1).reshape(-1).tolist()

    # mix_rate.sort()

    mask_lst = []
    mask_lst.append(generate_mask(torch.rand(num_patch, num_patch).cuda(), num_patch, mix_rate[0]))
    # process mask2: [1, 1, 0] [0, 1, 0] -> [1, 0, 0]
    for _ in range(1, num_mix - 1):
        new_mask = generate_mask(torch.rand(num_patch, num_patch).cuda(), num_patch, mix_rate[_])
        new_mask = torch.where(sum(mask_lst) == torch.ones_like(new_mask), torch.zeros_like(new_mask), new_mask)
        mask_lst.append(new_mask)
    # generate last mask
    last_mask = torch.ones_like(mask_lst[-1]) - sum(mask_lst)
    mask_lst.append(last_mask)
    # adjust mix_rate
    mix_rate = [mask.sum() / num_patch ** 2 for mask in mask_lst]
    img_index_lst = [torch.randperm(samples.shape[0]) for _ in range(num_mix)]

    num_batch, num_channel, img_size = samples.shape[0], samples.shape[1], samples.shape[2]
    patch_size = img_size // num_patch
    new_samples = samples.reshape(num_batch, num_channel, num_patch, patch_size, num_patch, patch_size)
    new_samples = sum([new_samples[img_index_lst[_]] * mask_lst[_].reshape(1, 1, num_patch, 1, num_patch, 1)  for _ in range(num_mix)])
    new_samples = new_samples.reshape(num_batch, num_channel, img_size, img_size)
    new_targets = sum([mix_rate[_] * targets[img_index_lst[_]] for _ in range(num_mix)])
    return new_samples, new_targets, ([targets[img_index_lst[_]] for _ in range(num_mix)], mask_lst)


def two_mix(samples, targets, num_patch=14, local_consist=7):
    # generate mask num_path * num_patch
    mask = torch.rand(num_patch, num_patch).cuda()
    lam = np.random.beta(1., 1.)
    mask = generate_mask(mask, num_patch, lam)
    #mask = torch.where(mask > torch.ones_like(mask) * (1 - mix_rate), torch.ones_like(mask), torch.zeros_like(mask))
    #mask = mask.reshape(num_patch // local_consist, local_consist, num_patch // local_consist, local_consist)
    #mask = mask[:, :1, :, :1].repeat(1, local_consist, 1, local_consist).reshape(num_patch, num_patch)
    
    mix_rate = mask.sum() / num_patch ** 2
    img_index = torch.randperm(samples.shape[0])


    num_batch, num_channel, img_size = samples.shape[0], samples.shape[1], samples.shape[2]
    patch_size = img_size // num_patch
    new_samples = samples.reshape(num_batch, num_channel, num_patch, patch_size, num_patch, patch_size)

    new_samples = new_samples * mask.reshape(1, 1, num_patch, 1, num_patch, 1) + new_samples[img_index] * (1 - mask.reshape(1, 1, num_patch, 1, num_patch, 1) )
    new_samples = new_samples.reshape(num_batch, num_channel, img_size, img_size)
    new_targets = targets * mix_rate + targets[img_index] * (1 - mix_rate)
    return new_samples, new_targets, ([targets, targets[img_index]], [mask, 1 - mask])


def repeat_two_mix(samples, targets, num_patch=14, repeat=2):
    mask = torch.rand(num_patch, num_patch).cuda()
    lam = np.random.beta(1., 1.)
    lam = max(max(lam, 1 - lam), 0.8)

    # lam = .5 # + np.random.randint(-10, 10) / 40. 

    mask = generate_mask(mask, num_patch, lam)
    mix_rate = mask.sum() / num_patch ** 2
    img_index = torch.randperm(samples.shape[0])
    repeat_index = torch.randperm(samples.shape[0])

    num_batch, num_channel, img_size = samples.shape[0], samples.shape[1], samples.shape[2]
    patch_size = img_size // num_patch
    new_samples = samples.reshape(num_batch, num_channel, num_patch, patch_size, num_patch, patch_size)
    repeat_samples = torch.cat((new_samples[repeat_index][:num_batch//2], new_samples[repeat_index][:num_batch//2]), dim=0)
    repeat_targets = torch.cat((targets[repeat_index][:num_batch//2], targets[repeat_index][:num_batch//2]), dim=0)

    new_samples = repeat_samples * mask.reshape(1, 1, num_patch, 1, num_patch, 1) + new_samples[img_index] * (1 - mask.reshape(1, 1, num_patch, 1, num_patch, 1) )
    new_samples = new_samples.reshape(num_batch, num_channel, img_size, img_size)
    new_targets = repeat_targets * mix_rate + targets[img_index] * (1 - mix_rate)
    return new_samples, new_targets, mix_rate, ([repeat_targets, targets[img_index]], [mask, 1 - mask], new_targets) # img_index)


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, teacher=None):
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
        
        samples, targets, aux_targets = repeat_two_mix(samples, targets, num_patch=samples.shape[-1] // 16)
        # samples, targets, aux_targets = multi_mix(samples, targets, num_patch=samples.shape[-1] // 16)

        with torch.cuda.amp.autocast():
            outputs, r_loss = model(samples, aux_targets)
            loss = criterion(samples, outputs, targets)

            loss_value = loss.item()
            loss += 1. * r_loss 

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
        metric_logger.meters['r_loss'].update(r_loss.item(), n=targets.shape[0])
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
    
    for images, target in metric_logger.log_every(data_loader, 100, header):
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
