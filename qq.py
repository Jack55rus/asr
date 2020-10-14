# coding: utf-8

import torch
from model.model import MyTransformer
data = torch.load("data.pt")
_, wav, ln, tin, tout = data
wav, tin, tout = wav.cuda(), tin.cuda(), tout.cuda()
model = MyTransformer(120, 37, 256, 1024)
model = model.cuda()
out = model(wav, tin)
tin_fake = torch.ones_like(tin)*36
tin_fake[:, :5] = tin[:, :5]
tin_fake[0, 0] = 1
# out_fake = model(wav, tin_fake))
out_fake = model(wav, tin_fake)
print((out - out_fake).max(dim=2))
