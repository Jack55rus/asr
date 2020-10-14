import yaml
from dataset import load_dataset
import torch
import torch.nn as nn
import time

from model.model import MyTransformer
from model.cer import Evaluater
from model.metrics import LabelSmoothingLoss
from torch.utils.tensorboard import SummaryWriter
from data.text import CharacterTextEncoder 

config = yaml.load(open("asr_russian.yaml", 'r'))

device = 'cuda'

def fetch_data(data):
    ''' Move data to device and compute text seq. length'''
    _, feat, feat_len, txt_in, txt_out = data
    feat = feat.to(device)
    feat_len = feat_len.to(device)
    txt_in = txt_in.to(device)
    txt_out = txt_out.to(device)
    txt_len = torch.sum(txt_out != 0, dim=-1)

    return feat, feat_len, txt_in, txt_out, txt_len



if __name__=="__main__":
    tr_set, dv_set, feat_dim, vocab_size, tokenizer, msg = \
            load_dataset(n_jobs = 8, use_gpu=True, pin_memory = True,
                         ascending=False, **config['data'])
    model = MyTransformer(sound_dim=120, text_dim=38, d_model=256, dim_feedforward=1024, dropout_rate=0.0)
    model = model.to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('./ckpt/dev_weights/Weight_299859.pth'))
    model.eval()

    tokinazer = CharacterTextEncoder.load_from_file(config['data']['text']['vocab_file'])
    txt_in = torch.ones(1, 80).type(torch.LongTensor).to(device)
    it = iter(dv_set)
    first = next(it)
    for j in range(5):
        txt_in = torch.ones(1, 150).type(torch.LongTensor).to(device)
        first = next(it)
        feat, _, _, txt_out, _ = fetch_data(first)

        i = 1
        char = -1
        s = time.time()
        while (char != 3):
            # feat, _, _, txt_out, _ = fetch_data(first)
            out = model(feat, txt_in[:, :i])
            seq = out.argmax(dim=-1)
            char = seq[:, i-1].item()
            # print(char, end='|')
            txt_in[:, i] = char
            i += 1
        e = time.time()
        # print('time elapsed:', e-s)
        print('Prediction', tokinazer.decode(txt_in[0, 1:50].cpu().detach().numpy()))
        print('True', tokinazer.decode(txt_out[0].cpu().detach().numpy()))