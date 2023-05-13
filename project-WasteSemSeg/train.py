import os
import random

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from model import ENet
from config import cfg
from loading_data import loading_data
from utils import *
from timer import Timer
from pytorchcv.model_provider import get_model as ptcv_get_model
import pdb

exp_name = cfg.TRAIN.EXP_NAME
log_txt = cfg.TRAIN.EXP_LOG_PATH + '/' + exp_name + '.txt'
writer = SummaryWriter(cfg.TRAIN.EXP_PATH+ '/' + exp_name)

pil_to_tensor = standard_transforms.ToTensor()
train_loader, val_loader, restore_transform = loading_data()

def main():

    print()

    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_txt, 'a') as f:
            f.write(''.join(cfg_lines) + '\n\n\n\n')
    if len(cfg.TRAIN.GPU_ID)==1:
        torch.cuda.set_device(cfg.TRAIN.GPU_ID[0])
    torch.backends.cudnn.benchmark = True

    net = []   
    
    if cfg.TRAIN.STAGE=='all':
        net = ptcv_get_model('icnet_resnetd50b_cityscapes', in_size=(224, 448), pretrained=False).eval().cuda()
        if cfg.TRAIN.PRETRAINED_ENCODER != '':
            encoder_weight = torch.load(cfg.TRAIN.PRETRAINED_ENCODER)
            del encoder_weight['classifier.bias']
            del encoder_weight['classifier.weight']
            # pdb.set_trace()
            net.encoder.load_state_dict(encoder_weight)
    elif cfg.TRAIN.STAGE =='encoder':
        net = ptcv_get_model('icnet_resnetd50b_cityscapes', in_size=(224, 448), pretrained=False).eval().cuda()

    if len(cfg.TRAIN.GPU_ID)>1:
        net = torch.nn.DataParallel(net, device_ids=cfg.TRAIN.GPU_ID).cuda()
    else:
        net=net.cuda()

    net.train()
    criterion = torch.nn.BCEWithLogitsLoss().cuda() # Binary Classification
    optimizer = optim.Adam(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.NUM_EPOCH_LR_DECAY, gamma=cfg.TRAIN.LR_DECAY)
    _t = {'train time' : Timer(),'val time' : Timer()} 
    
    # Validation
    mIoU_list = []
    validate(val_loader, net, criterion, optimizer, -1, restore_transform)

    print('\n')
   
    for epoch in range(cfg.TRAIN.MAX_EPOCH):

        _t['train time'].tic()
        train(train_loader, net, criterion, optimizer, epoch)
        _t['train time'].toc(average=False)
        print('🟠 TRAINING time of epoch {}/{} = {:.2f}s'.format(epoch, cfg.TRAIN.MAX_EPOCH ,_t['train time'].diff))
        _t['val time'].tic()
        mIoU = validate(val_loader, net, criterion, optimizer, epoch, restore_transform)
        mIoU_list.append(mIoU)
        validate(val_loader, net, criterion, optimizer, epoch, restore_transform)
        _t['val time'].toc(average=False)
        print('🟢 VALIDATION time of epoch {}/{} = {:.2f}s'.format(epoch, cfg.TRAIN.MAX_EPOCH, _t['val time'].diff))

    return mIoU_list

def train(train_loader, net, criterion, optimizer, epoch):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
   
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()


def validate(val_loader, net, criterion, optimizer, epoch, restore):
    net.eval()
    criterion.cpu()
    input_batches = []
    output_batches = []
    label_batches = []
    iou_ = 0.0
    for vi, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()

        # outputs is a tuple with 4 tensors inside -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
        '''
        tuple length = 4
        <class 'torch.Tensor'>
        torch.Size([16, 19, 224, 448])
        <class 'torch.Tensor'>
        torch.Size([16, 19, 56, 112])
        <class 'torch.Tensor'>
        torch.Size([16, 19, 28, 56])
        <class 'torch.Tensor'>
        torch.Size([16, 19, 14, 28])
        '''
        outputs = net(inputs)

        #DEBUG
        print(f'tuple length = {len(outputs)}')
        for tens in outputs:
            print(type(tens))
            print(tens.shape)
        print()
        print(f'labels: {type(labels)}, shape={labels.shape}')
        
        for lab in labels:
            print(type(lab))
            print(lab.shape)

        #for binary classification
        outputs[outputs>0.5] = 1
        outputs[outputs<=0.5] = 0
        #for multi-classification ???

        iou_ += calculate_mean_iu([outputs.squeeze_(1).data.cpu().numpy()], [labels.data.cpu().numpy()], 2)
    mean_iu = iou_/len(val_loader)   

    print('[mean IoU =  %.4f]' % (mean_iu)) 

    net.train()
    criterion.cuda()

    return mean_iu


if __name__ == '__main__':
    main()








