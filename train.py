# -*- coding: utf-8 -*-  

"""
Created on 2021/07/14

@author: Ruoyu Chen
"""

import argparse
import os
import numpy as np
import time

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from config import Config
from Logging import Logger
from dataset import Dataset

from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from Loss import MultiClassLoss

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

def define_backbone(network_name, num_classes, attribute_classes, strategy, pretrained=None):
    if network_name == "ResNet50":
        model = ResNet50(num_classes, attribute_classes, strategy)
    elif network_name == "ResNet101":
        model = ResNet101(num_classes, attribute_classes, strategy)

    if os.path.exists(pretrained):      # Load pretrained model
        model_dict = model.state_dict()
        pretrained_param = torch.load(pretrained)
        pretrained_dict = {k: v for k, v in pretrained_param.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        logger.write("\033[32m{} ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + 
                     "\033[0mLoad model" + 
                     "\033[34m {} ".format(network_name) + 
                     "\033[0mfrom pretrained" + 
                     "\033[34m {}".format(pretrained))
    else:               # Initialize from zero
        logger.write("\033[0mChoose network" + 
                     "\033[34m {} ".format(network_name) + 
                     "\033[0mas backbone.")
    return model

def define_Loss_function(loss_name, weight=None, pos_loss_weight=None):
    if loss_name == "Multi":
        Loss_function = MultiClassLoss()
    elif loss_name == "BCELoss":
        Loss_function = nn.BCEWithLogitsLoss(weight=weight,pos_weight=pos_loss_weight)
    return Loss_function

def define_optimizer(model, optimizer_name, learning_rate):
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr = learning_rate, weight_decay = 0.01)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0.01)
    return optimizer

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    device = torch.device("cuda")

    # Configuration file
    global cfg
    cfg = Config(args.configuration_file)

    # model save path
    model_save_path = os.path.join(cfg.CKPT_SAVE_PATH, 
        time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    mkdir(model_save_path)

    # logger
    global logger
    logger = Logger(os.path.join(model_save_path,"logging.log"))

    # Dataloader
    train_dataset = Dataset(dataset_root=cfg.DATASET_ROOT,dataset_list=cfg.DATASET_LIST_TRAIN,class_name=cfg.CLASS_NAME, strategy=cfg.STRATEGY, data_type="train")
    train_loader = DataLoader(train_dataset,batch_size=cfg.BATCH_SIZE,shuffle=True)

    model = define_backbone(cfg.BACKBONE, cfg.CLASS_NUM, cfg.ATTRIBUTE_NUM, cfg.STRATEGY ,cfg.PRETRAINED)

    # GPU
    if torch.cuda.is_available():
        model = model.cuda()
    # Multi GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Loss
    # weight = torch.Tensor([cfg.ATTR_LOSS_WEIGHT for i in range(cfg.BATCH_SIZE)]).to(device)
    weight = torch.Tensor(cfg.ATTR_LOSS_WEIGHT).to(device)
    Loss_function = define_Loss_function(cfg.LOSS_FUNCTION, weight, torch.Tensor(cfg.POS_LOSS_WEIGHT).to(device))

    # optimizer
    optimizer = define_optimizer(model, cfg.OPTIMIZER, cfg.LEARNING_RATE)

    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    for i in range(1,cfg.EPOCH+1):
        # scheduler.step()

        model.train()
        
        for ii, (data,label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = Loss_function(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i* len(train_loader) + ii

            if iters % 10 == 0:
                logger.write(
                    "\033[32m{} ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + 
                    "\033[0mtrain epoch " + 
                    "\033[34m{} ".format(i) +
                    "\033[0miter " + 
                    "\033[34m{} ".format(ii) + 
                    "\033[0mloss " + 
                    "\033[34m{}.".format(loss.item())
                )

        if i % 10 == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path,"backbone-item-epoch-"+str(i)+'.pth'))

def parse_args():
    parser = argparse.ArgumentParser(description='VOC 2008 datasets, attributes prediction')
    parser.add_argument('--configuration-file', type=str,
        default='./configs/Base-ResNet101-B.yaml',
        help='The model configuration file.')
    parser.add_argument('--gpu-device', type=str, default="0,1,2",
                        help='GPU device')
    # parser.add_argument('--LOG_txt', type=str, default="train.log",
    #                     help='log')
    args = parser.parse_args()

    return args
    
if __name__ == "__main__":
    args = parse_args()
    main(args)