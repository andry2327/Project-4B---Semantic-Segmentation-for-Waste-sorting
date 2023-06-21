import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import shutil
from config import cfg
import torch.nn.utils.prune as prune

import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150
plt.rcParams["figure.figsize"] = (10,6)

def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        #kaiming is first name of author whose last name is 'He' lol
        nn.init.kaiming_uniform(m.weight) 
        m.bias.data.zero_()

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially 
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def calculate_mean_iu(predictions, gts, num_classes):
    sum_iu = 0
    iu_classes = []
    for i in range(num_classes):
        n_ii = t_i = sum_n_ji = 1e-9
        for p, gt in zip(predictions, gts):
            n_ii += np.sum(gt[p == i] == i)
            t_i += np.sum(gt == i)
            sum_n_ji += np.sum(p == i)
        iou_i = float(n_ii) / (t_i + sum_n_ji - n_ii)
        sum_iu += iou_i
        iu_classes.append(iou_i)
    mean_iu = sum_iu / num_classes
    return mean_iu, iu_classes

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

def rmrf_mkdir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)

def rm_file(path_file):
    if os.path.exists(path_file):
        os.remove(path_file)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(cfg.VIS.PALETTE_LABEL_COLORS)

    return new_mask

#============================


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {'Overall Acc: \t': acc,
            'Mean Acc : \t': acc_cls,
            'FreqW Acc : \t': fwavacc,
            'Mean IoU : \t': mean_iu,}, cls_iu

# PLOTS UTILS
def showTicksLabels(xticks):
  if len(xticks) > 20:
    xticklabels = ['' if (int(i) % 5 != 0 and int(i) > 1) else str(int(i)) for i in xticks]
  else: xticklabels = xticks

  return xticklabels

def plot_mIoU_validation(net_str, mIoU_list, N_epoch, lr, N_classes):

    plt.figure(figsize=(10,5))

    plt.xlabel(f'epoch')
    plt.ylabel(f'mIoU')
    plt.title(f'{net_str} Validation')

    # plt.xticks([x+1 for x in range(N_epoch)])
    plt.plot([x+1 for x in range(N_epoch)], mIoU_list, marker='o')

    ax = plt.gca()

    ax.set_xticks([x+1 for x in range(N_epoch)])
    xticklabels = showTicksLabels([x+1 for x in range(N_epoch)])
    ax.set_xticklabels(xticklabels)

    plt.draw()

    fig_name = f'{net_str}__N_epoch={N_epoch}_LR={lr}_N_classes={N_classes}_->_MAXmIoU={round(max(mIoU_list), 4)}_LASTmIoU={round(mIoU_list[-1], 4)}'
    format = '.png'
    plt.savefig(fig_name+format, dpi=200)

    plt.show()

def load_checkpoints(net_name, net, optimizer):
    if len(os.listdir(f'checkpoints/{net_name}')) > 1:
            # load the saved checkpoint
            path_pth_file = [file for file in os.listdir(f'checkpoints/{net_name}') if '.pth' in file][0]
            checkpoint = torch.load(f'checkpoints/{net_name}/{path_pth_file}')

            # restore the state of the model and optimizer
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # resume training from the saved epoch
            start_epoch = checkpoint['epoch']

            # save previous mIoU list
            mIoU_list = checkpoint['mIoU_list']

            print(f"✅ Model '{path_pth_file}' Loaded\n")
            return net, optimizer, start_epoch, mIoU_list
    


# PRUNING AND QUANTIZATION

def recursive_module_name(module, method, amount): # 😢
    
    if isinstance(module, torch.nn):
        method(module, name='weight', amount=amount)
        is_pruned = torch.nn.utils.prune.is_pruned(module)
        print(f'pruned layer: {is_pruned}')
        return
    
    for _, module in module.named_modules():
        recursive_module_name(module, method, amount)
    
    return

def get_pruned_model(model, method=prune.random_unstructured, amount=0.8):

    parameters_to_prune = []
    ind = 0

    for name, module in model.named_modules():

        if isinstance(module, nn.Module):
            method(module, name='weight', amount=amount)
            is_pruned = torch.nn.utils.prune.is_pruned(module)
            print(f'pruned layer: {is_pruned}')

    '''  t = (module, name)
        parameters_to_prune.append(t)
    
    parameters_to_prune = tuple(parameters_to_prune)
    # DEBUG
    parameters_to_prune = parameters_to_prune[0]
    module_DEBUG = parameters_to_prune[0]

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=method,
        amount=amount
    )
    
    is_pruned = torch.nn.utils.prune.is_pruned(module_DEBUG)
    print(f'pruned: {is_pruned}')
    '''

    return model

