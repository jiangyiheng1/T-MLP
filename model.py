from modules import *


class C_MLP(nn.Module):
    def __init__(self, n_loc, n_gps, length, loc_dim, gps_dim, exp_factor, n_set, n_head, geo_depth, seq_depth, drop_ratio):
        super(C_MLP, self).__init__()
        '''Embedding Layer'''
        self.emb_loc = Embedding(n_loc, loc_dim)
        self.emb_gps = Embedding(n_gps, gps_dim)
        '''Geography Mixer'''
        self.geo_mixer_layer = Geo_Encoder_Layer(5, gps_dim, exp_factor, drop_ratio)
        self.geo_mixer = Geo_Encoder(self.geo_mixer_layer, geo_depth)
        '''Chunk Mixer'''
        dim = loc_dim + gps_dim
        self.mixer_layer = Encoder_Layer(length, dim, n_set, n_head, exp_factor, drop_ratio)
        self.mixer = Encoder(self.mixer_layer, seq_depth)

    def forward(self, src_loc, src_gps, trg_loc, trg_gps, data_size):
        # Embedding Locations
        src_loc_emb = self.emb_loc(src_loc)
        trg_loc_emb = self.emb_loc(trg_loc)
        # Embedding GPS code
        src_gps_emb = self.emb_gps(src_gps)
        trg_gps_emb = self.emb_gps(trg_gps)
        src_gps_emb = self.geo_mixer(src_gps_emb)
        trg_gps_emb = self.geo_mixer(trg_gps_emb)
        # Concat
        src = torch.cat([src_loc_emb, src_gps_emb], dim=-1)
        trg = torch.cat([trg_loc_emb, trg_gps_emb], dim=-1)
        # Mixer
        src = self.mixer(src)

        if self.training:
            src = src.repeat(1, trg.size(1)//src.size(1), 1)
        else:
            src = src[torch.arange(len(data_size)), torch.tensor(data_size) - 1, :]
            src = src.unsqueeze(1).repeat(1, trg.size(1), 1)
        output = torch.sum(src * trg, dim=-1)
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))