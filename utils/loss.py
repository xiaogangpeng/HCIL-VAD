import torch
import torch.nn.functional as F
import torch as th
from torch import nn
from utils.math_utils import artanh, tanh
eps = 1e-5

__all__ = ['InfoNCE', 'info_nce']


class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        #Cosine between positive pairs
        # positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)
        #
        # if negative_mode == 'unpaired':
        #     # Cosine between all query-negative combinations
        #     negative_logits = query @ transpose(negative_keys)
            
        
        
        # elif negative_mode == 'paired':
        #     query = query.unsqueeze(1)
        #     negative_logits = query @ transpose(negative_keys)
        #     negative_logits = negative_logits.squeeze(1)
        # print(f"query:{query.shape}  pos: {positive_key.shape}  neg :{negative_keys.shape}")
        # positive_logit = LorentzianDistance(query, positive_key).unsqueeze(-1)
        # negative_logits = LorentzianDistance(query, negative_keys).unsqueeze(-1)
        positive_logit = -PoincareDistance(query, positive_key, 0.5).unsqueeze(-1)
        negative_logits = -PoincareDistance(query, negative_keys, 0.5).unsqueeze(-1)
        # positive_logit = -ps(query, positive_key).unsqueeze(-1)
        # negative_logits = -ps(query, negative_keys).unsqueeze(-1)
        # First index in last dimension are the positive samples
        # print(f"pos : {positive_logit.shape}  neg:{negative_logits.shape}")
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        logits = logits / temperature
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

def LorentzianDistance(u, v):

    beta = 0.01
    u0 = th.sqrt(th.pow(u,2).sum(-1, keepdim=True) + beta)
    v0 = -th.sqrt(th.pow(v,2).sum(-1, keepdim=True) + beta)
    u = th.cat((u,u0), -1)
    v = th.cat((v,v0), -1)
    result = - 2 * beta - 2 * th.sum(u * v, dim=-1)
    return result

def PoincareDistance(p1, p2, c):
    sqrt_c = c ** 0.5
    dist_c = artanh(
        sqrt_c * mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
    )
    dist = dist_c * 2 / sqrt_c
    return dist ** 2

def mobius_add( x, y, c, dim=-1):
    min_norm = 1e-15
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(min_norm)


# class PoincareDistance(nn.Module):
#     def __int__(self):
#         super(PoincareDistance, self).__int__()

#     def forward(self, u, v):
#         boundary = 1 - eps
#         squnorm = th.clamp(th.sum(u * u, dim=-1), 0, boundary)
#         sqvnorm = th.clamp(th.sum(v * v, dim=-1), 0,boundary)
#         sqdist = th.sum(th.pow(u - v, 2), dim=-1)
#         x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
#         # arcosh
#         z = th.sqrt(th.pow(x, 2) - 1)
#         return th.log(x + z)


class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # if no margin assigned, use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor-pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor-neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)
        return loss

# def triplet_loss(anc, pos, neg, margin):
#     """
#     Triplet Loss的损失函数
#     """


#     # 欧式距离
#     pos_dist = torch.sum(torch.square(anc - pos), axis=-1, keepdims=True)
#     neg_dist = torch.sum(torch.square(anc - neg), axis=-1, keepdims=True)
#     basic_loss = pos_dist - neg_dist + torch.Tensor(margin).to(anc.device)
#     print(f"TeS:{basic_loss.shape}")
#     loss = torch.maximum(basic_loss, 0.0)

#     # print "[INFO] model - triplet_loss shape: %s" % str(loss.shape)
#     return loss

def triplet_loss(anchor, positive, negative, margin):
    pos_dist = PoincareDistance(anchor, positive, 0.5)
    neg_dist = PoincareDistance(anchor, negative, 0.5)
    # pos_dist = (anchor - positive).pow(2).sum(1) 
    # neg_dist = (anchor - negative).pow(2).sum(1) 
    loss = F.relu(pos_dist - neg_dist + margin) 
    return loss.mean()