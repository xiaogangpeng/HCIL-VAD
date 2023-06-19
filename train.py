import torch
import torch.nn as nn

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

# class AD_Loss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.bce = nn.BCELoss()
      
        
#     def forward(self, result, _label):
#         loss = {}

#         _label = _label.float()

#         triplet = result["triplet_margin"]
#         att = result['frame']
#         A_att = result["A_att"]
#         N_att = result["N_att"]
#         A_Natt = result["A_Natt"]
#         N_Aatt = result["N_Aatt"]
#         kl_loss = result["kl_loss"]
#         distance = result["distance"]
#         b = _label.size(0)//2
#         t = att.size(1)      
#         anomaly = torch.topk(att, t//16 + 1, dim=-1)[0].mean(-1)
#         anomaly_loss = self.bce(anomaly, _label)

#         panomaly = torch.topk(1 - N_Aatt, t//16 + 1, dim=-1)[0].mean(-1)
#         panomaly_loss = self.bce(panomaly, torch.ones((b)).cuda())
        
#         A_att = torch.topk(A_att, t//16 + 1, dim = -1)[0].mean(-1)
#         A_loss = self.bce(A_att, torch.ones((b)).cuda())

#         N_loss = self.bce(N_att, torch.ones_like((N_att)).cuda())    
#         A_Nloss = self.bce(A_Natt, torch.zeros_like((A_Natt)).cuda())

#         cost = anomaly_loss + 0.1 * (A_loss + panomaly_loss + N_loss + A_Nloss) + 0.1 * triplet + 0.001 * kl_loss + 0.0001 * distance

#         loss['total_loss'] = cost
#         loss['att_loss'] = anomaly_loss
#         loss['N_Aatt'] = panomaly_loss
#         loss['A_loss'] = A_loss
#         loss['N_loss'] = N_loss
#         loss['A_Nloss'] = A_Nloss
#         loss["triplet"] = triplet
#         loss['kl_loss'] = kl_loss
#         return cost, loss


criterion = torch.nn.BCELoss()


def train(net, normal_loader, abnormal_loader, optimizer, criterion, wind, index):
    net.train()
    net.flag = "Train"
    ninput, nlabel = next(normal_loader)
    ainput, alabel = next(abnormal_loader)
    inputs = torch.cat((ninput, ainput), 0).float().cuda()
    labels = torch.cat((nlabel, alabel), 0).float().cuda()
    seq_len = torch.sum(torch.max(torch.abs(inputs), dim=2)[0] > 0, 1)
    
    
    predict = net(inputs, seq_len)
    mil_logits = clas(predict['frame'], seq_len)
    loss = criterion(mil_logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # for key in loss.keys():     
    #     wind.plot_lines(key, loss[key].item())