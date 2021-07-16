# -*- coding: utf-8 -*-  

"""
Created on 2021/07/15

@author: Ruoyu Chen
"""

import argparse
import os
import numpy as np
import time

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader

from config import Config
from Logging import Logger
from dataset import Dataset

from prettytable import PrettyTable
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)

def Load_model(network_name, num_classes, attribute_classes, pretrained):
    if network_name == "ResNet101":
        model = ResNet101(num_classes, attribute_classes)

    assert os.path.exists(pretrained)
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

    return model 

def evaluation(model, validation_loader, attribute_num, attribute_name, device):
    """
    Evaluation the model in different outputs
    """
    model.eval()
    
    corrects = np.zeros(attribute_num)
    with torch.no_grad():
        for i, (data,labels) in enumerate(validation_loader):
            data = data.to(device)
            labels = labels.to(device).long()

            outputs = model(data)
            
            ii = 0
            for output, label in zip(outputs, labels.t()):
                # output: Torch_size(batch,33)
                # label: Torch_size(batch)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = pred.eq(label.view_as(pred)).sum().item()

                corrects[ii] += correct
                ii+=1
    corrects /= len(validation_loader.dataset)

    acc_in_each_attribute(corrects, attribute_name)
    return None

def acc_in_each_attribute(corrects, attribute_name):
    """
    Generate the evaluation table
    """
    assert len(corrects)==len(attribute_name)
    
    table = PrettyTable(["Attribute Name", "Accuracy"])
    
    for i in range(len(corrects)):
        table.add_row([str(i)+". "+attribute_name[i],
                       "{:.2f}%".format(corrects[i]*100)])
    print(table)
    return None

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    device = torch.device("cuda")

    # Configuration file
    cfg = Config(args.configuration_file)

    # model save path
    results_save_path = os.path.join(cfg.EVALUATION_SAVE_PATH, 
        time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    mkdir(results_save_path)

    # logger
    global logger
    logger = Logger(os.path.join(results_save_path, "logging.log"))

    # Dataloader
    validation_dataset = Dataset(dataset_root=cfg.DATASET_ROOT,dataset_list=cfg.DATASET_LIST_TRAIN,class_name=cfg.CLASS_NAME)
    validation_loader = DataLoader(validation_dataset,batch_size=cfg.BATCH_SIZE,shuffle=True)

    model = Load_model(cfg.BACKBONE, cfg.CLASS_NUM, cfg.ATTRIBUTE_NUM, args.Test_model)

    # GPU
    if torch.cuda.is_available():
        model = model.cuda()
    # Multi GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    evaluation(model, validation_loader, cfg.ATTRIBUTE_NUM, cfg.ATTRIBUTE_NAME, device)

def parse_args():
    parser = argparse.ArgumentParser(description='VOC 2008 datasets, attributes prediction')
    parser.add_argument('--configuration-file', type=str,
        default='./configs/Base-ResNet101.yaml',
        help='The model configuration file.')
    parser.add_argument('--gpu-device', type=str, default="0,1,2,3",
                        help='GPU device')
    parser.add_argument('--Test-model', type=str, 
    # default="./checkpoint/backbone-item-epoch-990.pth",
    default="./checkpoint/2021-07-16-12:15:47/backbone-item-epoch-300.pth",
                        help='Model weight for testing.')
    args = parser.parse_args()

    return args
    
if __name__ == "__main__":
    args = parse_args()
    main(args)