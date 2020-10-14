import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from model.module import PositionalEncoding, Conv2dSubsampling, LinearWithPosEmbedding

def get_enc_padding_mask(tensor):
    return torch.sum(tensor, dim=-1).eq(0)

def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class MyTransformer(nn.Module):
    def __init__(self, sound_dim, text_dim, d_model, dim_feedforward, dropout_rate=0.0, device="cuda"):
        super(MyTransformer,self).__init__()

        self.device = device

        self.sound_embed = Conv2dSubsampling(sound_dim, d_model, dropout_rate)
        # self.sound_embed = nn.Linear(sound_dim, d_model)
        self.text_embed = nn.Embedding(text_dim, d_model, )
        self.pos_encoder = PositionalEncoding(d_model , dropout_rate)

        self.transformer = nn.Transformer(d_model, nhead= 8, num_encoder_layers= 12,\
                                          num_decoder_layers = 6, dim_feedforward = dim_feedforward,\
                                          dropout = dropout_rate, activation = 'gelu')

        # self.lin_ctc = nn.Linear(d_model, text_dim + 1)

        self.out_lin = nn.Linear(d_model, text_dim)
        # self.softmax = nn.Softmax(dim=2)

    def forward(self, sound, target):
        # print(sound.shape, target.shape)
        enc_mask = get_enc_padding_mask(sound).to(self.device)
        sound, enc_mask = self.sound_embed(sound, enc_mask)
        sound[enc_mask] = 0
        
        target = self.text_embed(target)
        target = self.pos_encoder(target)
        trg_mask = generate_square_subsequent_mask(target.size(1)).to(self.device)

        # out = self.transformer(sound.permute(1,0,2), target.permute(1, 0, 2),\
        #                        tgt_mask=trg_mask, src_key_padding_mask=enc_mask)

        enc = self.transformer.encoder(sound.permute(1, 0, 2), src_key_padding_mask=enc_mask)
        # out_ctc = 
        # print('enc', enc.shape)
        # input('stop model')
        out = self.transformer.decoder(target.permute(1, 0, 2), enc, tgt_mask=trg_mask)

        out = self.out_lin(out.permute(1, 0, 2))

        # out_ctc = self.out_lin(enc)
        
        return out
