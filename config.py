# -*- coding: utf-8 -*-  

"""
Created on 2021/07/14

@author: Ruoyu Chen
"""

import yaml

class Config():
    """
    Contain the hyber parameters
    """
    def __init__(self,yaml_config):
        dict = self._init_dict(yaml_config)
        self.VERSION = dict["VERSION"]
        self.BACKBONE = dict["BACKBONE"]
        self.PRETRAINED = dict["PRETRAINED"]

        self.BATCH_SIZE = dict["BATCH_SIZE"]
        self.OPTIMIZER = dict["OPTIMIZER"]
        self.LEARNING_RATE = dict["LEARNING_RATE"]

        self.EPOCH = dict["EPOCH"]

        # self.GPU_DEVICE = dict["GPU_DEVICE"]

        self.DATASET_ROOT = dict["DATASET_ROOT"]
        self.DATASET_LIST_TRAIN = dict["DATASET_LIST_TRAIN"]
        self.DATASET_LIST_TEST = dict["DATASET_LIST_TEST"]
        self.ATTRIBUTE_NUM = dict["ATTRIBUTE_NUM"]
        self.CLASS_NUM = dict["CLASS_NUM"]
        self.CLASS_NAME = dict["CLASS_NAME"]
        self.ATTRIBUTE_NAME = dict["ATTRIBUTE_NAME"]

        self.CKPT_SAVE_PATH = dict["CKPT_SAVE_PATH"]
        self.EVALUATION_SAVE_PATH = dict["EVALUATION_SAVE_PATH"]

    def _init_dict(self, yaml_config):
        dict = yaml.load(open(yaml_config), Loader=yaml.FullLoader)
        print("Load the configuration file from " + 
              "\033[35m{}".format(yaml_config))
        return dict
