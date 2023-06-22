"""Graph encoders."""


import manifolds
import layers.hyp_layers as hyp_layers
from utils.pre_utils import *
import layers.lorentz_layers as loren_layers

class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c
    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output


class HGCN(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, args, manifold, gcn_in_feat):
        super(HGCN, self).__init__(c)
        self.manifold = getattr(manifolds, manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args, gcn_in_feat)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att, args.local_agg
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True


    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HGCN, self).encode(x_hyp, adj)

class HyboNet(Encoder):
    """
    HyboNet.
    """

    def __init__(self, c, args, manifold, gcn_in_feat):
        super(HyboNet, self).__init__(c)
        self.manifold = getattr(manifolds, manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args, gcn_in_feat)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.LorentzGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, args.bias, args.dropout, act, args.use_att, args.local_agg, nonlin=act if i != 0 else None
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        # x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        # x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        # x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HyboNet, self).encode(x, adj)


class LGCN(Encoder):
    def __init__(self, c, args):
        super(LGCN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = loren_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        lgnn_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            in_dim = in_dim - 1 if i != 0 else in_dim   # for layer more than 2
            act = acts[i]
            lgnn_layers.append(
                loren_layers.LorentzGraphNeuralNetwork(
                    self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att
                )
            )
        self.layers = nn.Sequential(*lgnn_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        # print(f'lorentz tangent {x.shape}')
        x_loren = self.manifold.normalize_input(x, c=self.curvatures[0])
        return super(LGCN, self).encode(x_loren, adj)

    def reset_parameters(self):
        for tmp_layer in self.layers:
            tmp_layer.reset_parameters()


