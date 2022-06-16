""" This file builds model for the framework """
import logging
from copy import deepcopy

from models import  resnet, wideresnet
from models.resnet-mlmi import get_resnet18

models = {
    "wideresnet": wideresnet.build,
    # "resnext": resnext.build,
    "resnet50": resnet.resnet50,
    "resnet18": resnet.resnet18
}


def build(cfg, logger=None):

    # init params
    init_params = deepcopy(cfg)
    type_name = init_params.pop("type")
    
    if type_name == "resnet18mlmi":
        model = get_resnet18()
        if logger is not None:
            logger.info("{} Total params: {:.2f}M".format(
                type_name, sum(p.numel() for p in model.parameters()) / 1e6))
        else:
            logging.info("{} Total params: {:.2f}M".format(
                type_name, sum(p.numel() for p in model.parameters()) / 1e6))
        return model
        
    
    

    # init model
    model = models[type_name](**init_params)
    if logger is not None:
        logger.info("{} Total params: {:.2f}M".format(
            type_name, sum(p.numel() for p in model.parameters()) / 1e6))
    else:
        logging.info("{} Total params: {:.2f}M".format(
            type_name, sum(p.numel() for p in model.parameters()) / 1e6))
    return model
