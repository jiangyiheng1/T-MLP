import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce


class Embedding(nn.Module):
    def __init__(self, vocab_size, num_units, zeros_pad=True, scale=True):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale
        self.lookup_table = nn.Parameter(torch.Tensor(vocab_size, num_units))
        nn.init.xavier_normal_(self.lookup_table.data)
        if self.zeros_pad:
            self.lookup_table.data[0, :].fill_(0)

    def forward(self, inputs):
        if self.zeros_pad:
            self.padding_idx = 0
        else:
            self.padding_idx = -1
        outputs = F.embedding(inputs, self.lookup_table, self.padding_idx, None, 2, False, False)
        if self.scale:
            outputs = outputs * (self.num_units ** 0.5)
        return outputs


class Chunk_MLP(nn.Module):
    def __init__(self, length, features, num_sets, num_heads, drop_ratio):
        super(Chunk_MLP, self).__init__()
        assert length % num_sets == 0 and features % num_heads == 0
        self.s = length // num_sets
        self.h = features // num_heads
        global_mixer = torch.triu(torch.ones(length, length))
        self.global_mixer = torch.nn.parameter.Parameter(global_mixer, requires_grad=True)
        local_mixer = torch.triu(torch.ones(self.s, self.s))
        local_mixer = repeat(local_mixer, 's1 s2 -> n_set n_head s1 s2', n_set=num_sets, n_head=num_heads)
        self.local_mixer = torch.nn.parameter.Parameter(local_mixer, requires_grad=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x):
        x = rearrange(x, 'b (n_set s) (n_head h) -> b n_set n_head h s', h=self.h, s=self.s)
        x = torch.matmul(x, self.local_mixer)
        x = self.act(x)
        x = self.dropout(x)
        x = rearrange(x, 'b n_set n_head h s -> b (n_head h) (n_set s)')
        x = torch.matmul(x, self.global_mixer)
        x = self.act(x)
        x = self.dropout(x)
        x = rearrange(x, 'b d n -> b n d')
        return x


class Feed_Forward(nn.Module):
    def __init__(self, features, exp_factor, drop_ratio):
        super(Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(features, exp_factor * features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(exp_factor * features, features)
        self.drop = nn.Dropout(drop_ratio)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Encoder_Layer(nn.Module):
    def __init__(self, length, features, num_sets, num_heads, exp_factor, drop_ratio):
        super(Encoder_Layer, self).__init__()
        self.chunk_mlp = Chunk_MLP(length, features, num_sets, num_heads, drop_ratio)
        self.feed_forward = Feed_Forward(features, exp_factor, drop_ratio)
        self.norm_1 = nn.LayerNorm(features)
        self.norm_2 = nn.LayerNorm(features)

    def forward(self, x):
        x = x + self.chunk_mlp(self.norm_1(x))
        x = x + self.feed_forward(self.norm_2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, layer, depth):
        super(Encoder, self).__init__()
        self.encoder = nn.ModuleList([layer for i in range(depth)])

    def forward(self, x):
        for i, layer in enumerate(self.encoder):
            x = layer(x)
        return x


class Geo_MLP(nn.Module):
    def __init__(self, length, drop_ratio):
        super(Geo_MLP, self).__init__()
        self.geo_mixer = nn.Linear(length, length)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x):
        x = rearrange(x, 'b n l d -> b n d l')
        x = self.geo_mixer(x)
        x = self.act(x)
        x = self.dropout(x)
        x = rearrange(x, 'b n d l -> b n l d')
        return x


class Geo_Encoder_Layer(nn.Module):
    def __init__(self, length, features, exp_factor, drop_ratio):
        super(Geo_Encoder_Layer, self).__init__()
        self.geo_mlp = Geo_MLP(length, drop_ratio)
        self.feed_forward = Feed_Forward(features, exp_factor, drop_ratio)
        self.norm_1 = nn.LayerNorm(features)
        self.norm_2 = nn.LayerNorm(features)

    def forward(self, x):
        x = x + self.geo_mlp(self.norm_1(x))
        x = x + self.feed_forward(self.norm_2(x))
        return x


class Geo_Encoder(nn.Module):
    def __init__(self, layer, depth):
        super(Geo_Encoder, self).__init__()
        self.encoder = nn.ModuleList([layer for i in range(depth)])

    def forward(self, x):
        for i, layer in enumerate(self.encoder):
            x = layer(x)
        x = reduce(x, 'b n l d -> b n d', 'mean')
        return x