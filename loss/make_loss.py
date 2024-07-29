# make_loss.py

"""
Visual grounding - Text image attention, transformer based
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
import wandb

MARGIN = 0.5
MARGIN_LOSS_WEIGHT = 0.05
TAU = 0.99
LAMBDA_ENTROPY = 0.005

UNCERTAINITIES = 0

class MarginLoss(nn.Module):
    #TODO: Calculate the uncertainities count correctly
    def __init__(self, tau=0.5, lambda_entropy=0.1):
        super(MarginLoss, self).__init__()
        self.tau = tau
        self.lambda_entropy = lambda_entropy

    def forward(self, logits, uncertianties):
        if isinstance(logits, list):
            margin_losses = [self.calculate_margin_loss(logit, uncertianties) for logit in logits]
            margin_loss, uncertainties = zip(*margin_losses)
            margin_loss = list(margin_loss)
            uncertainties = list(uncertainties)

            return sum(margin_loss) / len(margin_loss), sum(uncertainties) / len(uncertainties)
        else:
            return self.calculate_margin_loss(logits, uncertianties)

    def calculate_margin_loss(self, logits, uncertianties):
        probs = F.softmax(logits, dim=1)
        known_probs = torch.sum(probs[:, :-1], dim=1)
        unknown_prob = probs[:, -1]
        margin = known_probs - unknown_prob
        uncertianties += (margin < self.tau).sum().item()

        # Calculate entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean()

        wandb.log({"entropy": entropy})
        wandb.log({"margin": margin})
        margin_loss = torch.mean(F.relu(self.tau - margin))
        total_loss = margin_loss + self.lambda_entropy * entropy

        return total_loss, uncertianties

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

    margin_loss = MarginLoss(tau=TAU, lambda_entropy=LAMBDA_ENTROPY)
    print(f"\nTAU: {TAU}\nMARGIN: {MARGIN}\nMARGIN_LOSS_WEIGHT: {MARGIN_LOSS_WEIGHT}\nLAMBDA_ENTROPY: {LAMBDA_ENTROPY}\n")

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, i2tscore=None, uncertianties=0):
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

                    margin_loss_value, uncertainties_count = margin_loss(score, uncertianties)
                    wandb.log({"Margin Loss Value": margin_loss_value, "loss value": loss, "Uncertainities count": uncertainties_count})
                    loss += MARGIN_LOSS_WEIGHT * margin_loss_value
                    wandb.log({"loss + margin (weighted)": loss})
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

                    margin_loss_value, uncertainties_count = margin_loss(score, uncertianties)
                    wandb.log({"Margin Loss Value": margin_loss_value, "loss value": loss, "Uncertainities count": uncertainties_count})
                    loss += MARGIN_LOSS_WEIGHT * margin_loss_value

                    return loss
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion
