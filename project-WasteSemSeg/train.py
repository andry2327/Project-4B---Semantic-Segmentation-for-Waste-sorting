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
from bisenet import BiSeNetV2
from pytorchcv.model_provider import get_model as ptcv_get_model

from config import cfg
from loading_data import loading_data
from utils import *
from timer import Timer
import pdb
from tqdm import tqdm

exp_name = cfg.TRAIN.EXP_NAME
log_txt = cfg.TRAIN.EXP_LOG_PATH + '/' + exp_name + '.txt'
writer = SummaryWriter(cfg.TRAIN.EXP_PATH+ '/' + exp_name)

pil_to_tensor = standard_transforms.ToTensor()
train_loader, val_loader, restore_transform = loading_data()

def set_net(net_name):
    net_name = net_name.lower()
    if(net_name == 'enet'):
        if cfg.TRAIN.STAGE=='all':
            net = ENet(only_encode=False)
            if cfg.TRAIN.PRETRAINED_ENCODER != '':
                encoder_weight = torch.load(cfg.TRAIN.PRETRAINED_ENCODER)
                del encoder_weight['classifier.bias']
                del encoder_weight['classifier.weight']
                # pdb.set_trace()
                net.encoder.load_state_dict(encoder_weight)
        elif cfg.TRAIN.STAGE =='encoder':
            net = ENet(only_encode=True)
    elif (net_name == 'bisenet'):
        net = BiSeNetV2(n_classes=cfg.DATA.NUM_CLASSES)
    else : 
        net =  ptcv_get_model('icnet_resnetd50b_cityscapes', in_size=(224, 448), num_classes=cfg.DATA.NUM_CLASSES, pretrained=False, aux=False).eval().cuda()
    return net

def main(net_name = 'Enet', checkpoint = False):

    net_name = net_name.lower()

    save_every = 10
    start_epoch = 0

    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_txt, 'a') as f:
            f.write(''.join(cfg_lines) + '\n\n\n\n')
    if len(cfg.TRAIN.GPU_ID)==1:
        torch.cuda.set_device(cfg.TRAIN.GPU_ID[0])
    torch.backends.cudnn.benchmark = True

    #net=[]
    net = set_net(net_name)    

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

    if checkpoint:
        net, optimizer, start_epoch, mIoU_list = load_checkpoints(net_name, net, optimizer)
        #start_epoch += 1 #because the start_epoch was already trained.

    print()
    print(f'Initial mIoU NO TRAINING: ', end='')

    validate(val_loader, net, criterion, optimizer, -1, restore_transform)

    print('\n')
   
    for epoch in range(start_epoch, start_epoch+cfg.TRAIN.MAX_EPOCH):

        _t['train time'].tic()
        train(train_loader, net, criterion, optimizer, epoch)
        _t['train time'].toc(average=False)
        print('🟠 TRAINING time of epoch {}/{} = {:.2f}s'.format(epoch+1, start_epoch+cfg.TRAIN.MAX_EPOCH, _t['train time'].diff))
        _t['val time'].tic()
        mIoU = validate(val_loader, net, criterion, optimizer, epoch, restore_transform)
        mIoU_list.append(mIoU)
        _t['val time'].toc(average=False)
        print('🟢 VALIDATION time of epoch {}/{} = {:.2f}s'.format(epoch+1, start_epoch+cfg.TRAIN.MAX_EPOCH,  _t['val time'].diff))

        # save the model state every few epochs 
        if (epoch+1) % save_every == 0:
            checkpoint = {
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch+1,
                'mIoU_list': mIoU_list
            }
            torch.save(checkpoint, f'checkpoints/{net_name}/checkpoint_{net_name}_epoch={epoch+1}.pth')
            print(f"🔷 Model checkpoint '{f'checkpoint_{net_name}_epoch={epoch+1}.pth'}' saved")
            if epoch >= start_epoch+save_every:
                os.remove(f'checkpoints/{net_name}/checkpoint_{net_name}_epoch={epoch+1-save_every}.pth')

    return mIoU_list


def train(train_loader, net, criterion, optimizer, epoch):

    train_progress = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1} Training", leave=False)

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
   
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()

        train_progress.update(1)
    
    train_progress.close()


def validate(val_loader, net, criterion, optimizer, epoch, restore):
    net.eval()
    criterion.cpu()
    input_batches = []
    output_batches = []
    label_batches = []
    iou_ = 0.0
    validation_progress = tqdm(total=len(val_loader), desc=f"Epoch {epoch+1} Validation", leave=False)
    for vi, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()
        outputs = net(inputs)
        #for binary classification
        # outputs[outputs>0.5] = 1
        # outputs[outputs<=0.5] = 0
        
        # multi-classification
        outputs[outputs<=0.5] = 0 #background
        outputs[(outputs>0.5) & (outputs<=1.5)] = 1 #aluminium
        outputs[(outputs>1.5) & (outputs<=2.5)] = 2 #paper
        outputs[(outputs>2.5) & (outputs<=3.5)] = 3 #bottle
        outputs[outputs>3.5] = 4 #nylon


        iou_ += calculate_mean_iu([outputs.squeeze_(1).data.cpu().numpy()], [labels.data.cpu().numpy()], cfg.DATA.NUM_CLASSES+1)
        validation_progress.update(1)
    
    validation_progress.close()
    mean_iu = iou_/len(val_loader)

    print('[mean IoU =  %.4f]' % (mean_iu)) 

    net.train()
    criterion.cuda()

    return mean_iu


if __name__ == '__main__':
    main()








