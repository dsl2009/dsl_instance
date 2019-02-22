import torch
from torch import nn
from loss.loss_utils import *


def loss_function(heatmap_pred, heatmap_target, regre_pred, regre_target, mask):
    heatmap_pred = sigmoid(heatmap_pred)
    heat_loss = neg_loss(heatmap_pred, heatmap_target)
    reger_loss = regr_loss(regre_pred, regre_target, mask)
    return heatmap_pred, heat_loss+reger_loss

