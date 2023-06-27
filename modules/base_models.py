"""Base model class."""

import torch
import torch.nn as nn
import manifolds
import modules.encoders as encoders


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args, model, manifold):
        super(BaseModel, self).__init__()
        self.manifold_name = manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))

        self.manifold = getattr(manifolds, self.manifold_name)()

        in_feat = args.gcn_in_feat
        out_feat = args.gcn_hidden
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            in_feat = in_feat + 1
            # out_feat = out_feat + 1

        self.encoder = getattr(encoders, model)(self.c, args, manifold, in_feat, out_feat)


    def encode(self, x, adj):

        if self.manifold_name in ['Lorentz', 'Hyperboloid']:
            o = torch.zeros_like(x)
            x = torch.cat([o[:, :, 0:1], x], dim=-1)
            if self.manifold_name == 'Lorentz':
                x = self.manifold.expmap0(x)

        h = self.encoder.encode(x, adj)
        return h


class FHyperGCN(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args, model, manifold):
        super(FHyperGCN, self).__init__(args, model, manifold)

    def decode(self, h, adj):
        raise

