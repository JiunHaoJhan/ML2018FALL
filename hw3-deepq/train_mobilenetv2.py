# 2018 HTC Corporation. All Rights Reserved.
#
# This source code is licensed under the HTC license which can be found in the
# LICENSE file in the root directory of this work.

import torch
import PIL
import random
from torchvision import transforms

def get_random_seed():
    """
    this function is called once before the training starts

    returns:
        an integer as the random seed, or None for random initialization
    """
    seed = None
    return seed

def get_model_spec():
    """
    this function is called once for setting up the model

    returns:
        a dictionary contains following items:
            model_name: one of the 'resnet50' and 'mobilenetv2'
            pretrained: a boolean value indicating whether to use pre-trained model
                        if False is returned, default initialization will be used
    """
    return {"model_name": "mobilenetv2", "pretrained": True}

def get_optimizer(params):
    """
    this function is called once for setting up the optimizer

    args:
        params: the set of the parameters to be optimized
                should be passed to torch.optim.Optimizer

    returns:
        an torch.optim.Optimizer optimizing the given params

    notes:
        don't modify params
    """
    #res
    #optimizer = torch.optim.SGD(params, lr=0.1, weight_decay=0)
    #mob
    optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.001)

    return optimizer

def get_eval_spec():
    """
    this function is called once for setting up evaluation / inferencing

    returns:
        a dictionary contains following items:
            transform: a transform used to preprocess evalutaion / inferencing images
                       should be a callable which takes a PIL image of 3 x 256 x 256
                       and produce either a 3 x 244 x 244 tensor or a NC x 3 x 244 x 244 tensor
                        in the latter case:
                            NC stands for the number of crops of an image,
                            NC predictions will be inferenced on the NC crops
                            then those predictions will be average to produce a final prediction
            batchsize: an integer between 1 and 64
    """
    transform =  transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
        ])
    return {"transform": transform, "batchsize": 32}


def before_epoch(train_history, validation_history):
    """
    this function is called before every training epoch

    args:
        train_history:
            a 3 dimensional python list (i.e. list of list of list)
            the j-th element in i-th list is a list containing two entries,
            stands for the [accuracy, loss] for the j-th batch in the i-th epoch

            len(train_history) indicate the index of current epoch
            this value should be within the range of 1~50

        validation_history:
            a 2 dimentsional python list (i.e. list of list)
            the i-th element is a list containing two entry,
            stands for the [accuracy, loss] for the validation result of the i-th epoch

    returns:
        a dictionary contains following items:
            transform: a transform used to preprocess training images
                       should be a callable which takes a PIL image of 3 x 256 x 256
                       and produces a 3 x 224 x 224 tensor
            batchsize: an integer between 1 and 64
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(180),
        transforms.RandomAffine(30, translate=(0.2, 0.2)),
        transforms.RandomApply([transforms.Grayscale(3)], p=0.3),
        transforms.Resize(224),
        transforms.ToTensor()
        ])
    #aug
    n_epoch = len(train_history)
    batchsize=32
    if n_epoch % 10 ==0 and n_epoch!=50:
        batchsize=batchsize + 8*(n_epoch//10)
    return {"transform": transform, "batchsize": batchsize}

def before_batch(train_history, validation_history):
    """
    this function is called before each training batch

    args: please refer to before_epoch()

    returns:
        a dictionary contains the following items:
            optimizer: a dictionary of optimizer hyperparameters
            batch_norm: a float stands for the value of momentum in all batch normalization layers
                        or None indicating no changes should be made
            drop_out: a float stands for the value of drop out probability
                      or None indicating no changes should be made

    notes:
        drop_out should always be None when using resnet50 since there are no dropout layers in it
    """
    n_epoch = len(train_history)
    lr_rate = 0.001
    if n_epoch % 20 == 0:
        lr_rate = lr_rate / 10**(n_epoch//20)
    return {"optimizer": {"lr": lr_rate}, "batch_norm": 0.5, "drop_out":0.5}

def save_model_as(train_history, validation_history):
    """
    this function is called after each epoch's training
    the returned value will be used to determine whether to save the model at this point or not

    args: please refer to before_epoch()

    returns:
        a string, the filename as which the model is going to be saved
        or None indicating no saving is desired for this epoch
    """
    n_epoch = len(train_history)
    max_val = 0 
    for i in range(len(validation_history)):
        if validation_history[i][0] > max_val:
            max_val = validation_history[i][0]
            max_epoch = i
    if (n_epoch-1) == max_epoch:
        return str(n_epoch-1) +'_save_mob.pth.tar'
    else:
        return None

