from modules.base_models import *
from layers.hyp_layers import *
from geoopt import ManifoldParameter
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
import numpy as np
import torch.nn.functional as F
import math
from torch.nn.modules.module import Module
from torch import FloatTensor
from torch.nn.parameter import Parameter
import manifolds

class HypClassifier(nn.Module):
    """
    Hyperbolic Classifier
    """

    def __init__(self, args):
        super(HypClassifier, self).__init__()
        # self.manifold = getattr(manifolds, args.manifold1)()
        self.input_dim = args.gcn_hidden * 4
        self.output_dim = args.num_classes
        self.use_bias = args.bias
        # if self.manifold.name == "Lorentz":         
        #     self.cls = ManifoldParameter(self.manifold.random_normal((args.num_classes, self.input_dim), std=1./math.sqrt(self.input_dim)), manifold=self.manifold)
        # else:
        self.cls = nn.Linear(self.input_dim, self.output_dim)
        if args.bias:
            self.bias = nn.Parameter(torch.zeros(args.num_classes))

    def forward(self, x):
        # if self.manifold.name == "Lorentz":      
        #     return (2 + 2 * self.manifold.cinner(x, self.cls)) + self.bias
        # else:
        return self.cls(x)


class DistanceAdj(Module):

    def __init__(self):
        super(DistanceAdj, self).__init__()
        self.sigma = Parameter(FloatTensor(1))
        self.sigma.data.fill_(0.1)

    def forward(self, batch_size, max_seqlen, device):
        # To support batch operations
        self.arith = np.arange(max_seqlen).reshape(-1, 1)
        dist = pdist(self.arith, metric='cityblock').astype(np.float32)
        self.dist = torch.from_numpy(squareform(dist)).to(device)
        self.dist = torch.exp(-self.dist / torch.exp(torch.tensor(1.)))
        self.dist = torch.unsqueeze(self.dist, 0).repeat(batch_size, 1, 1).to(device)
        return self.dist



class Model(nn.Module):
    def __init__(self, args, flag):
        super(Model, self).__init__()
        self.flag=flag
        self.args = args
        # self.manifold = getattr(manifolds, args.manifold1)()
        # if self.manifold.name in ['Lorentz', 'Lorentzian','Hyperboloid']:
        #     args.gcn_in_feat =  args.gcn_in_feat + 1

        self.disAdj = DistanceAdj()

        self.conv1d1 = nn.Conv1d(in_channels=args.c_in_feat, out_channels=args.c_hidden, kernel_size=1, padding=0)
        self.conv1d2 = nn.Conv1d(in_channels=args.c_hidden, out_channels=args.c_out_feat, kernel_size=1, padding=0)

        self.HFSGCN1 = FHyperGCN(args, args.model1, args.manifold1)
        self.HTRGCN1 = FHyperGCN(args, args.model1, args.manifold1)

        self.HFSGCN2 = FHyperGCN(args, args.model2, args.manifold2)
        self.HTRGCN2 = FHyperGCN(args, args.model2, args.manifold2)

        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.HyperCLS = HypClassifier(args)
        self.args = args

        self.encoder_n = nn.Sequential(nn.Linear(args.gcn_hidden*2, args.gcn_hidden*2))
        self.encoder_a = nn.Sequential(nn.Linear(args.gcn_hidden*2, args.gcn_hidden*2))



    def forward(self, inputs, seq_len):
        x = inputs
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1
        
        x = x.permute(0, 2, 1)  # for conv1d
        x = self.relu(self.conv1d1(x))
        x = self.dropout(x)
        x = self.relu(self.conv1d2(x))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # b*t*c

        disadj = self.disAdj(x.shape[0], x.shape[1], x.device).to(x.device)
        adj = self.adj2(x, seq_len)
        
        # proj_x = self.expm(x, self.args.manifold1)
        # proj_y = self.expm(x, self.args.manifold2)
        proj_x = x
        proj_y = x

        x1 = self.relu(self.HFSGCN1.encode(proj_x, adj))
        x1 = self.dropout(x1)
        x2 = self.relu(self.HTRGCN1.encode(proj_x, disadj))
        x2 = self.dropout(x2)
        out_x = torch.cat((x1, x2), 2)


        x3 = self.relu(self.HFSGCN2.encode(proj_y, adj))
        x3 = self.dropout(x3)
        x4 = self.relu(self.HTRGCN2.encode(proj_y, disadj))
        x4 = self.dropout(x4)
        out_y = torch.cat((x3, x4), 2)


        o = torch.zeros_like(out_y)
        out_y = torch.cat([o[:, :, 0:1], out_y], dim=-1)
        out_y = self.HFSGCN2.manifold.lorentz_to_poincare(out_y)
        out = torch.cat((out_x, out_y), 2)
                 
        frame_prob = self.HyperCLS(out).reshape((b, n, -1)).mean(1)
        
        if self.flag=='Train':
            batch_size = int(b//2)
            poin_nor_out = out_x[:batch_size]
            poin_abn_out = out_x[batch_size:]
            lore_nor_out = out_y[:batch_size]
            lore_abn_out = out_y[batch_size:]
            nor_score = frame_prob[:batch_size]
            abn_score = frame_prob[batch_size:]
            # nor_out_aug = self.encoder_n(nor_out)
            # abn_out_aug = self.encoder_a(abn_out)

            return{
                "frame": frame_prob,
                'poin_nor_feat': poin_nor_out,
                'poin_abn_feat': poin_abn_out,
                'lore_nor_feat': lore_nor_out,
                'lore_abn_feat': lore_abn_out,
                'abn_score': abn_score,
                'nor_score': nor_score
                }     
        else:
             return{
                "frame": frame_prob
                }

    # def expm(self, x, manifold):
    #     manifold_ = getattr(manifolds, manifold)()
    #     if manifold_.name in ['Lorentz', 'Hyperboloid']:
    #         o = torch.zeros_like(x)
    #         x = torch.cat([o[:, :, 0:1], x], dim=-1)
    #         if manifold_.name == 'Lorentz':
    #             x = manifold_.expmap0(x)
    #         return x
    #     else:
    #         return x

    def adj(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = self.lorentz_similarity(x, x, self.manifold.k)
        x2 = torch.exp(-x2)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.8, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.8, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2
        return output

    def adj2(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1))  # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2 / (x_norm_x + 1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.8, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.8, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output



    def lorentz_similarity(self, x: torch.Tensor, y: torch.Tensor, k) -> torch.Tensor:
        '''
        d = <x, y>   lorentz metric
        '''
        self.eps = {torch.float32: 1e-6, torch.float64: 1e-8}
        idx = np.concatenate((np.array([-1]), np.ones(x.shape[-1] - 1)))
        diag = torch.from_numpy(np.diag(idx).astype(np.float32)).to(x.device)
        temp = x @ diag
        xy_inner = -(temp @ y.transpose(-1, -2))
        xy_inner_ = F.threshold(xy_inner, 1, 1)
        sqrt_k = k**0.5
        dist = sqrt_k * self.arccosh(xy_inner_ / k)
        dist = torch.clamp(dist, min=self.eps[x.dtype], max=200)
        return dist

    def arccosh(self, x):
        """
        Element-wise arcosh operation.
        Parameters
        ---
        x : torch.Tensor[]
        Returns
        ---
        torch.Tensor[]
            arcosh result.
        """
        return torch.log(x + torch.sqrt(torch.pow(x, 2) - 1))

