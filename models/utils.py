import math
import logging
import torch
from torch.nn import ELU, ReLU, Sigmoid, Linear, Sequential, Module, Parameter, Softmax
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

USE_CUDA = torch.cuda.is_available()

pi = math.pi

# ==NN Utilities== #
ACT_FNS = ['ELU', 'ReLU', 'Sigmoid']

log_lvls = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.DEBUG,
    'CRITICAL': logging.CRITICAL
}


def linear_stack(layers, activation='ReLU'):
    assert activation in ACT_FNS
    if activation == 'ELU':
        activation = ELU
    elif activation == 'ReLU':
        activation = ReLU
    elif activation == 'Sigmoid':
        activation = Sigmoid

    net = []
    for i in range(len(layers) - 1):
        net.append(Linear(layers[i], layers[i + 1]))
        if i < len(layers) - 2:
            net.append(activation())
    return Sequential(*net)


def make_transformer(d_model, nhead, num_layers):
    encoder_layer = TransformerEncoderLayer(d_model, nhead)
    return TransformerEncoder(encoder_layer, num_layers=num_layers)


class GlobalAttentionHead(Module):

    def __init__(self, inp_dim, out_dim, query_dim=None):

        super().__init__()

        assert isinstance(inp_dim, int) and inp_dim > 0
        self.inp_dim = inp_dim

        assert isinstance(out_dim, int) and out_dim > 0
        self.out_dim = out_dim

        if query_dim is None:
            query_dim = max(inp_dim, out_dim)

        assert isinstance(query_dim, int) and query_dim > 0
        self.query_dim = query_dim

        self.global_query_weights = Parameter(torch.ones(self.query_dim),
                                              requires_grad=True)
        self.key_weights = Linear(self.inp_dim, self.query_dim)
        self.value_weights = Linear(self.inp_dim, self.out_dim)
        # self.sftmx = Softmax()

    def forward(self, embs, lengths):

        # Expects padded sequence with batch first False
        assert issubclass(type(embs), torch.Tensor)
        assert embs.dim() == 3 and embs.size(-1) == self.inp_dim

        # Get batch size
        bs = len(lengths)

        # Create output tensor
        outs = torch.zeros(bs, self.out_dim)
        if embs.device.type == 'cuda':
            outs = outs.cuda()

        for b in range(bs):
            b_seq = embs[:lengths[b].int().item(), b, :]
            b_keys = self.key_weights(b_seq)
            b_vals = self.value_weights(b_seq)

            b_attn_energies = b_keys.matmul(
                self.global_query_weights.unsqueeze(1))
            b_weights = b_attn_energies.exp()
            b_weights = b_weights / b_weights.sum()
            # self.sftmx(b_attn_energies.transpose(0, 1),
            #                       dim=lengths[b].int().item())

            outs[b] = b_vals.transpose(0, 1).matmul(
                b_weights).transpose(0, 1)

        return outs
