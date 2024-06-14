# make_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
import wandb

MARGIN = 0.5
MARGIN_LOSS_WEIGHT = 6
TAU = 0.1
UNCERTAINITIES = 0

class MarginLoss(nn.Module):
    #TODO: Calculate the uncertainities count correctly
    def __init__(self, tau=0.5):
        super(MarginLoss, self).__init__()
        self.tau = tau

    def forward(self, logits, uncertianties):
        print("shape of logits", len(logits), logits[0].shape, logits[1].shape)
        if isinstance(logits, list):
            margin_losses = [self.calculate_margin_loss(logit, uncertianties) for logit in logits]
            margin_loss, uncertainties = zip(*margin_losses)
            margin_loss = list(margin_loss)
            uncertainties = list(uncertainties)

            print("margin losses", margin_losses, margin_losses[0])
            return sum(margin_loss) / len(margin_loss), sum(uncertainties) / len(uncertainties)
        else:
            return self.calculate_margin_loss(logits, uncertianties)

    def calculate_margin_loss(self, logits, uncertianties):
        probs = F.softmax(logits, dim=1)
        print("\n PROBS are ", probs)
        print("\n SUM OR PROBS", torch.sum(probs[0]))
        known_probs, _ = torch.max(probs[:, :-1], dim=1)
        unknown_prob = probs[:, -1]
        margin = known_probs - unknown_prob
        uncertianties += (margin < self.tau).sum().item()

        # print("\nUnknow Probabilities", unknown_prob)
        # print("\n\nKnown Probabilities", known_probs)
        wandb.log({"margin": margin})
        margin_loss = torch.mean(F.relu(self.tau - margin))
        # if margin_loss > 0:
        #     # self.uncertainities_count += 1
        #     wandb.log({"Uncertainities count": uncertianties, "Unknow Probs": unknown_prob, "Known Probs": known_probs})
        return margin_loss, uncertianties

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
                        print("I2T Loss", I2TLOSS)

                    # print(type(score), type(target))
                    margin_loss_value, uncertainties_count = margin_loss(score, uncertianties)
                    wandb.log({"Margin Loss Value": margin_loss_value, "loss value": loss, "Uncertainities count": uncertainties_count})
                    print("Margin Loss", margin_loss_value)
                    loss += MARGIN_LOSS_WEIGHT * margin_loss_value
                    wandb.log({"loss + margin (weighted)": loss})
                    #TODO: Return margin loss along with the loss, make sure that the loss is combined so that loss.backward() works and you'll get magin loss per iteration, get the anomalies per iteration
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
