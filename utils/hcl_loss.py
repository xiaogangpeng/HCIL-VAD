import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .InfoNCE import InfoNCE


def HCIL(abn_logits, nor_logits, seq_len, poin_abn_feat, poin_nor_feat, lore_abn_feat, lore_nor_feat, device, batch_size):
    abn_rep = torch.zeros(0).to(device)
    nor_rep = torch.zeros(0).to(device)
    bgd_rep = torch.zeros(0).to(device)
    abn_seq_len = seq_len[batch_size:]
    nor_seq_len = seq_len[:batch_size]

    for i in range(abn_logits.size(0)):
        cur_nor_topk, cur_nor_topk_indices = torch.topk(nor_logits[i][:nor_seq_len[i]], k=int(torch.div(nor_seq_len[i], 16, rounding_mode='floor') + 1), largest=True)
        cur_nor_rep_topk = nor_feat[i][cur_nor_topk_indices]
        cur_dim = cur_nor_rep_topk.size()
        cur_nor_rep_topk = torch.mean(cur_nor_rep_topk, 0, keepdim=True).expand(cur_dim)
        nor_rep = torch.cat((nor_rep, cur_nor_rep_topk), 0)

        # bgd features
        cur_nor_inverse_topk, cur_nor_inverse_topk_indices = torch.topk(nor_logits[i][:nor_seq_len[i]], k=int(torch.div(nor_seq_len[i], 16, rounding_mode='floor') + 1), largest=False)     # return k min score and indices
        cur_nor_inverse_rep_topk = nor_feat[i][cur_nor_inverse_topk_indices]   #  get min k value
        bgd_rep = torch.cat((bgd_rep, cur_nor_inverse_rep_topk), 0)

        cur_abn_topk, cur_abn_topk_indices = torch.topk(abn_logits[i][:abn_seq_len[i]], k=int(torch.div(abn_seq_len[i], 16, rounding_mode='floor') + 1), largest=True)
        cur_abn_rep_topk = abn_feat[i][cur_abn_topk_indices]
        cur_dim = cur_abn_rep_topk.size()
        cur_abn_rep_topk = torch.mean(cur_abn_rep_topk, 0, keepdim=True).expand(cur_dim)
        abn_rep = torch.cat((abn_rep, cur_abn_rep_topk), 0)


    min_len, max_len = min(len(abn_rep), len(nor_rep)), max(len(abn_rep), len(nor_rep))
    idx = random.sample(range(0, max_len), min_len)
    if len(abn_rep) > len(nor_rep):
        abn_rep = abn_rep[idx]
    else:
        nor_rep = nor_rep[idx]
        bgd_rep = bgd_rep[idx]
        
    hcil = InfoNCE(negative_mode='unpaired')
    if nor_rep.size(0) == 0 or abn_rep.size(0) == 0:
        return 0.0
    else:
        loss_a2n = hcil(abn_rep, nor_rep, bgd_rep)
        return loss_a2n



# def HCIL(logits1, logits2, seq_len, feat1, feat2, device, batch_size):
#     abn_rep1 = torch.zeros(0).to(device)
#     nor_rep = torch.zeros(0).to(device)
#     abn_rep2 = torch.zeros(0).to(device)
#     logits1, logits2 = torch.sigmoid(logits1), torch.sigmoid(logits2)

#     nor_seq_len, abn_seq_len = seq_len[:batch_size], seq_len[batch_size:]
#     nor_logits1, nor_logits2 = logits1[:batch_size].squeeze(), logits2[:batch_size].squeeze()
#     abn_logits1, abn_logits2 = logits1[batch_size:],  logits2[batch_size:]
#     nor_feat1, nor_feat2 = feat1[:batch_size], feat2[:batch_size]
#     abn_feat1, abn_feat2 = feat1[batch_size:], feat2[batch_size:]
    

#     for i in range(abn_logits1.size(0)): 

#         cur_nor_topk, cur_nor_topk_indices = torch.topk(nor_logits1[i][:nor_seq_len[i]], k=int(torch.div(nor_seq_len[i], 16, rounding_mode='floor') + 1), largest=True)
#         cur_nor_rep_topk = nor_feat1[cur_nor_topk_indices]
#         cur_dim = cur_nor_rep_topk.size()
#         cur_nor_rep_topk = torch.mean(cur_nor_rep_topk, 0, keepdim=True).expand(cur_dim)
#         nor_rep = torch.cat((nor_rep, cur_nor_rep_topk), 0)

#         # bgd features
#         cur_nor_topk, cur_nor_topk_indices = torch.topk(nor_logits2[i][:nor_seq_len[i]], k=int(torch.div(nor_seq_len[i], 16, rounding_mode='floor') + 1), largest=True)
#         cur_nor_rep_topk = nor_feat2[cur_nor_topk_indices]
#         cur_dim = cur_nor_rep_topk.size()
#         cur_nor_rep_topk = torch.mean(cur_nor_rep_topk, 0, keepdim=True).expand(cur_dim)
#         nor_rep = torch.cat((nor_rep, cur_nor_rep_topk), 0)

          
#         cur_abn_topk3, cur_abn_topk_indices3 = torch.topk(abn_logits1[i][:abn_seq_len[i]], k=int(torch.div(abn_seq_len[i], 16, rounding_mode='floor') + 1), largest=True)
#         cur_abn_rep_topk3 = abn_feat1[i][cur_abn_topk_indices3]
#         cur_dim3 = cur_abn_rep_topk3.size()
#         cur_abn_rep_topk3 = torch.mean(cur_abn_rep_topk3, 0, keepdim=True).expand(cur_dim3)
#         abn_rep1 = torch.cat((abn_rep1, cur_abn_rep_topk3), 0)

#         cur_abn_topk4, cur_abn_topk_indices4 = torch.topk(abn_logits2[i][:abn_seq_len[i]], k=int(torch.div(abn_seq_len[i], 16, rounding_mode='floor') + 1), largest=True)
#         cur_abn_rep_topk4 = abn_feat2[i][cur_abn_topk_indices4]
#         cur_dim4 = cur_abn_rep_topk4.size()
#         cur_abn_rep_topk4 = torch.mean(cur_abn_rep_topk4, 0, keepdim=True).expand(cur_dim4)
#         abn_rep2 = torch.cat((abn_rep2, cur_abn_rep_topk4), 0)

#     print(f"tesT: {abn_rep1.shape} {abn_rep2.shape}  {nor_rep.shape}")
#     min_len, max_len = min(len(abn_rep), len(nor_rep)), max(len(abn_rep), len(nor_rep))
#     idx = random.sample(range(0, max_len), min_len)
#     if len(abn_rep) > len(nor_rep):
#         abn_rep = abn_rep[idx]
#     else:
#         nor_rep = nor_rep[idx]
#         bgd_rep = bgd_rep[idx]
        
#     hcil = InfoNCE(negative_mode='unpaired')
#     if nor_rep.size(0) == 0 or abn_rep.size(0) == 0:
#         return 0.0
#     else:
#         loss_a2n = hcil(abn_rep, nor_rep, bgd_rep)
#         return loss_a2n