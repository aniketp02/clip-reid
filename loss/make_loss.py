# make_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
import wandb

MARGIN = 0.5
MARGIN_LOSS_WEIGHT = 0.8
TAU = 0.8

class MarginLoss(nn.Module):
    def __init__(self, tau=0.5):
        super(MarginLoss, self).__init__()
        self.tau = tau

    def forward(self, logits, target):
        if isinstance(logits, list):
            margin_losses = [self.calculate_margin_loss(logit) for logit in logits]
            return sum(margin_losses) / len(margin_losses)
        else:
            return self.calculate_margin_loss(logits)

    def calculate_margin_loss(self, logits):
        probs = F.softmax(logits, dim=1)
        known_probs, _ = torch.max(probs[:, :-1], dim=1)
        unknown_prob = probs[:, -1]
        margin = known_probs - unknown_prob
        margin_loss = torch.mean(F.relu(self.tau - margin))
        return margin_loss

def make_loss(cfg, num_classes):    
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    margin_loss = MarginLoss(tau=TAU)  

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, i2tscore=None):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS)
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                    if i2tscore is not None:
                        I2TLOSS = xent(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss

                    # print(type(score), type(target))
                    margin_loss_value = margin_loss(score, target)
                    wandb.log({"Margin Loss": margin_loss_value, "loss value": loss})
                    print("Margin Loss", margin_loss_value)
                    loss += MARGIN_LOSS_WEIGHT * margin_loss_value

                    return loss
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS)
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                    if i2tscore is not None:
                        I2TLOSS = F.cross_entropy(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss

                    margin_loss_value = margin_loss(score, target)
                    wandb.log({"Margin Loss": margin_loss_value, "loss value": loss})
                    print("Margin Loss", margin_loss_value)
                    loss += MARGIN_LOSS_WEIGHT * margin_loss_value

                    return loss
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion
