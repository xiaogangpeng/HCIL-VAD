import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
from utils.InfoNCE import InfoNCE

def norm(data):
    l2=torch.norm(data, p = 2, dim = -1, keepdim = True)
    return torch.div(data, l2)


def clas(logits, seq_len):
    logits = logits.squeeze()
    instance_logits = torch.zeros(0).to(logits.device)  # tensor([])
    for i in range(logits.shape[0]):
        if seq_len is None:
            tmp = torch.mean(logits[i]).view(1)
        else:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(torch.div(seq_len[i], 16, rounding_mode='floor') + 1),
                            largest=True)
            tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat((instance_logits, tmp))
    instance_logits = torch.sigmoid(instance_logits)
    return instance_logits



class HCIL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hcil_loss = InfoNCE(negative_mode='unpaired')

    def forward(self, abn_score, nor_score, poin_abn_feat, poin_nor_feat, lore_abn_feat, lore_nor_feat, seq_len, batch_size, device):       
        abn_rep1 = torch.zeros(0).to(device)
        abn_rep2 = torch.zeros(0).to(device)
        nor_rep = torch.zeros(0).to(device)
        abn_seq_len = seq_len[batch_size:]
        nor_seq_len = seq_len[:batch_size]
        abn_logits = torch.sigmoid(abn_score)
        nor_logits = torch.sigmoid(nor_score)


        for i in range(abn_logits.size(0)):
            cur_nor_topk, cur_nor_topk_indices = torch.topk(nor_logits[i][:nor_seq_len[i]], k=int(torch.div(nor_seq_len[i], 16, rounding_mode='floor') + 1), largest=True)
            cur_nor_rep_topk = poin_nor_feat[i][cur_nor_topk_indices]
            cur_dim = cur_nor_rep_topk.size()
            cur_nor_rep_topk = torch.mean(cur_nor_rep_topk, 0, keepdim=True).expand(cur_dim)
            nor_rep = torch.cat((nor_rep, cur_nor_rep_topk), 0)

            # bgd features
            cur_nor_topk, cur_nor_topk_indices = torch.topk(nor_logits[i][:nor_seq_len[i]], k=int(torch.div(nor_seq_len[i], 16, rounding_mode='floor') + 1), largest=True)
            cur_nor_rep_topk = lore_nor_feat[i][cur_nor_topk_indices]
            cur_dim = cur_nor_rep_topk.size()
            cur_nor_rep_topk = torch.mean(cur_nor_rep_topk, 0, keepdim=True).expand(cur_dim)
            nor_rep = torch.cat((nor_rep, cur_nor_rep_topk), 0)



            # cur_nor_inverse_topk, cur_nor_inverse_topk_indices = torch.topk(nor_logits[i][:nor_seq_len[i]], k=int(torch.div(nor_seq_len[i], 16, rounding_mode='floor') + 1), largest=False)     # return k min score and indices
            # cur_nor_inverse_rep_topk = nor_feat[i][cur_nor_inverse_topk_indices]   #  get min k value
            # bgd_rep = torch.cat((bgd_rep, cur_nor_inverse_rep_topk), 0)

            cur_abn_topk, cur_abn_topk_indices = torch.topk(abn_logits[i][:abn_seq_len[i]], k=int(torch.div(abn_seq_len[i], 16, rounding_mode='floor') + 1), largest=True)
            cur_abn_rep_topk = poin_abn_feat[i][cur_abn_topk_indices]
            cur_dim = cur_abn_rep_topk.size()
            cur_abn_rep_topk = torch.mean(cur_abn_rep_topk, 0, keepdim=True).expand(cur_dim)
            abn_rep1 = torch.cat((abn_rep1, cur_abn_rep_topk), 0)

            cur_abn_topk, cur_abn_topk_indices = torch.topk(abn_logits[i][:abn_seq_len[i]], k=int(torch.div(abn_seq_len[i], 16, rounding_mode='floor') + 1), largest=True)
            cur_abn_rep_topk = lore_abn_feat[i][cur_abn_topk_indices]
            cur_dim = cur_abn_rep_topk.size()
            cur_abn_rep_topk = torch.mean(cur_abn_rep_topk, 0, keepdim=True).expand(cur_dim)
            abn_rep2 = torch.cat((abn_rep2, cur_abn_rep_topk), 0)


        min_len, max_len = min(len(abn_rep1), len(nor_rep)), max(len(abn_rep1), len(nor_rep))
        idx = random.sample(range(0, max_len), min_len)
        if len(abn_rep1) > len(nor_rep):
            abn_rep1 = abn_rep1[idx]
            abn_rep2 = abn_rep2[idx]
        else:
            nor_rep = nor_rep[idx]
            
        if nor_rep.size(0) == 0 or abn_rep1.size(0) == 0 or abn_rep2.size(0) == 0:
            return 0.0
        else:
            loss_a2n = self.hcil_loss(abn_rep1, abn_rep2, nor_rep)
            return loss_a2n
            
    # def forward(self, result, _label):
    #     loss = {}

    #     _label = _label.float()

    #     triplet = result["triplet_margin"]
    #     att = result['frame']
    #     A_att = result["A_att"]
    #     N_att = result["N_att"]
    #     A_Natt = result["A_Natt"]
    #     N_Aatt = result["N_Aatt"]
    #     kl_loss = result["kl_loss"]
    #     distance = result["distance"]
    #     b = _label.size(0)//2
    #     t = att.size(1)      
    #     anomaly = torch.topk(att, t//16 + 1, dim=-1)[0].mean(-1)
    #     anomaly_loss = self.bce(anomaly, _label)

    #     panomaly = torch.topk(1 - N_Aatt, t//16 + 1, dim=-1)[0].mean(-1)
    #     panomaly_loss = self.bce(panomaly, torch.ones((b)).cuda())
        
    #     A_att = torch.topk(A_att, t//16 + 1, dim = -1)[0].mean(-1)
    #     A_loss = self.bce(A_att, torch.ones((b)).cuda())

    #     N_loss = self.bce(N_att, torch.ones_like((N_att)).cuda())    
    #     A_Nloss = self.bce(A_Natt, torch.zeros_like((A_Natt)).cuda())

    #     cost = anomaly_loss + 0.1 * (A_loss + panomaly_loss + N_loss + A_Nloss) + 0.1 * triplet + 0.001 * kl_loss + 0.0001 * distance

    #     loss['total_loss'] = cost
    #     loss['att_loss'] = anomaly_loss
    #     loss['N_Aatt'] = panomaly_loss
    #     loss['A_loss'] = A_loss
    #     loss['N_loss'] = N_loss
    #     loss['A_Nloss'] = A_Nloss
    #     loss["triplet"] = triplet
    #     loss['kl_loss'] = kl_loss
    #     return cost, loss


criterion = torch.nn.BCELoss()
HCIL_Loss = HCIL()

def train(net, normal_loader, abnormal_loader, optimizer, criterion, wind, index):
    loss = {}
    net.train()
    net.flag = "Train"
    ninput, nlabel = next(normal_loader)
    ainput, alabel = next(abnormal_loader)
    inputs = torch.cat((ninput, ainput), 0).float().cuda()
    labels = torch.cat((nlabel, alabel), 0).float().cuda()
    seq_len = torch.sum(torch.max(torch.abs(inputs), dim=2)[0] > 0, 1)


    predict = net(inputs, seq_len)
    mil_logits = clas(predict['frame'], seq_len)
    mil_loss = criterion(mil_logits, labels)

    hcil_loss = HCIL_Loss(predict['abn_score'],
                          predict['nor_score'],
                          predict['poin_nor_feat'],
                          predict['poin_abn_feat'],
                          predict['lore_nor_feat'],
                          predict['lore_abn_feat'],
                          seq_len,
                          ninput.size(0),
                          inputs.device
                          )

    cost = mil_loss + 0.05 * hcil_loss

    # cost = mil_loss
    
    loss['total_loss'] = cost
    loss['mil_loss'] = mil_loss
    loss['hcil_loss'] = hcil_loss
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    for key in loss.keys():     
        wind.plot_lines(key, loss[key].item())