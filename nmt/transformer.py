'''Borrowed from The Annotated Transformer'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from tensortorch import easytt as ttm
from tensortorch import tucker


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N, compress_k=0,  compressed_layer=None):
        super(Encoder, self).__init__()
        if compress_k > 0:
            self.layers = clones(compressed_layer, compress_k)
            if N-compress_k > 0:
                new_layers = clones(layer, N - compress_k)
                for l in new_layers:
                    self.layers.append(l)
        else:
            self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N, compress_k=0,  compressed_layer=None):
        super(Decoder, self).__init__()
        if compress_k > 0:
            self.layers = clones(compressed_layer, compress_k)
            if N-compress_k > 0:
                new_layers = clones(layer, N - compress_k)
                for l in new_layers:
                    self.layers.append(l)
        else:
            self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, compress=True):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        d_in_modes = [2, 4, 8, 4, 2]
        tt_ranks = [1, 2, 2, 2, 2, 1]
        is_bias = False

        if compress:
            self.linears = clones(tucker.TuckerLinear(d_in_modes, d_in_modes, [2] * len(d_in_modes), bias=False), 4)
        else:
            self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1, compress_mode=None):
        super(PositionwiseFeedForward, self).__init__()

        # TODO create module with modes parameters
        d_out_modes = [4, 4, 8, 4, 4]
        d_in_modes = [2, 4, 8, 4, 2]

        if compress_mode is not None:
            if compress_mode == 'tt':
                # tensor train

                tt_ranks = [1, 2, 2, 2, 2, 1]
                tt_ranks2 = [ 1,  1, 16, 54,  6,  1]
                tt_ranks3 = [1, 4, 4, 4, 4, 1]
                tt_ranks4 = [1, 2, 4, 4, 2, 1]
                tt_ranks8 = [1, 2, 4, 8, 4, 2, 1]

                self.w_1 = ttm.TTLayer(d_in_modes, d_out_modes, tt_ranks3, bias=False)
                self.w_2 = ttm.TTLayer(d_out_modes,d_in_modes, tt_ranks4, bias=False)
            else:
                # tucker
                tucker_ranks = [2] * len(d_in_modes)
                
                self.w_1 = tucker.TuckerLinear(d_in_modes, d_out_modes, tucker_ranks, bias=False)
                self.w_2 = tucker.TuckerLinear(d_out_modes, d_in_modes, tucker_ranks, bias=False)
        else:
            self.w_1 = nn.Linear(d_model, d_ff)
            self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.tensor(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, \
               compress=False, compress_mode='tt', compress_att=False, num_compress_enc=6, num_compress_dec=6):
    "Helper: Construct a model from hyperparameters."
    enc_k = dec_k= 0
    enc_comp_ff = dec_comp_ff= None

    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, compress=compress_att)
    if compress:
        enc_k = num_compress_enc
        dec_k = num_compress_dec
        enc_comp_ff = PositionwiseFeedForward(d_model, d_ff, dropout, compress_mode=compress_mode)
        dec_comp_ff = PositionwiseFeedForward(d_model, d_ff, dropout, compress_mode=compress_mode)
        enc_ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        dec_ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    else:
        enc_ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        dec_ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    # Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout),N, \
    #             compress_k=6, compressed_layers=EncoderLayer(d_model, c(attn), c(comp_ff), dropout))
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(enc_ff), dropout), N, \
                            compress_k=enc_k, compressed_layer=\
                    EncoderLayer(d_model, c(attn), c(enc_comp_ff ), dropout)),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(dec_ff), dropout), N, \
                compress_k=dec_k, compressed_layer= \
                    DecoderLayer(d_model, c(attn),c(attn), c(dec_comp_ff), dropout)),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
