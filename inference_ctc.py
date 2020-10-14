import yaml
from dataset import load_dataset
import torch
import torch.nn as nn
import time

from model.model_ctc import MyTransformer
from model.cer import Evaluater
from model.metrics import LabelSmoothingLoss
from torch.utils.tensorboard import SummaryWriter
from data.text import CharacterTextEncoder 
from copy import deepcopy
import editdistance

config = yaml.load(open("asr_russian.yaml", 'r'))

device = 'cuda'

def fetch_data(data):
    ''' Move data to device and compute text seq. length'''
    _, feat, feat_len, txt_in, txt_out, _ = data
    feat = feat.to(device)
    feat_len = feat_len.to(device)
    txt_in = txt_in.to(device)
    txt_out = txt_out.to(device)
    txt_len = torch.sum(txt_out != 0, dim=-1)

    return feat, feat_len, txt_in, txt_out, txt_len


def evaluate(model, dv_set, eval_stop_iter, device, config, write_txt = False, wer=False):
    # it = iter(dv_set)
    model = deepcopy(model)
    model.eval()
    # model = model.module
    evaluater = Evaluater(config['data']['text']['vocab_file'])
    cer = 0
    wer = 0
    total_cer = 0
    # total_time = time.time()
    file = open('preds.txt', 'w')
    if eval_stop_iter > len(dv_set):
        eval_stop_iter = len(dv_set)
    with torch.no_grad():
        for j, batch_eval in enumerate(dv_set):
            if j % 10 == 0:
                print('eval iter:', j)
            if j == eval_stop_iter:
                eval_stop_iter = j
                break
            # batch_eval = next(it)
            feat, _, _, txt_out, txt_len = fetch_data(batch_eval)
            max_len = max(txt_len)
            txt_in = torch.ones(feat.shape[0], max_len).type(torch.LongTensor).to(device)
            i = 1
            char = -1
            enc_out, new_feat_len = model.encoder_step(feat)
            out1 = model.ctc_step(enc_out)
            out1 = nn.Softmax(dim=2)(out1)
            seq1 = out1.argmax(dim=2)
            # print('seq:', seq1)
            # while (i < max_len):
            #     out = model.decoder_step(enc_out, txt_in[:, :i])
            #     out = nn.Softmax(dim=2)(out)
            #     seq = out.argmax(dim=2)
            #     char = seq[:, i-1]
            #     txt_in[:, i] = char
            #     i += 1
            if not write_txt:
                # print('dec:', txt_in[:, 1:])
                # input()
                if j == 0:
                    # cer, pred_sent, true_sent = evaluater.evaluate(txt_in[:, 1:], txt_out, visual=True, visual_batch=True)
                    cer, pred_sent, true_sent = evaluater.evaluate(seq1, txt_out, visual=True, visual_batch=True)
                    # input()
                else:
                    # cer, pred_sent, true_sent = evaluater.evaluate(txt_in[:, 1:], txt_out, visual=False, visual_batch=True)
                    cer, pred_sent, true_sent = evaluater.evaluate(seq1, txt_out, visual=False, visual_batch=True)
                    # print('Pred:', pred_sent, 'True:', true_sent)
                    # wer += calc_wer(pred_sent, true_sent)
                    # print('WER:', wer)
            else:
                # cer, pred_sent, true_sent = evaluater.evaluate(txt_in[:, 1:], txt_out, visual=False, visual_batch=True)
                cer, pred_sent, true_sent = evaluater.evaluate(seq1, txt_out, visual=False, visual_batch=True)
                # print('Pred:', pred_sent, 'True:', true_sent)
                for i in range(len(pred_sent)):
                    file.write('Pred:' + '\n' + pred_sent[i] + '\n')
                    file.write('True:' + '\n' + true_sent[i] + '\n')
                    file.write('--'*20)
                    # print('Pred:', pred_sent, 'True:', true_sent)
            # cer = evaluater.evaluate(txt_in[:, 1:], txt_out, visual=True, visual_batch=False)
            total_cer += cer
            # print('CER:', total_cer)
    print('total wer', wer)
    file.write('total cer =' + str(round(total_cer / eval_stop_iter, 3)))
    file.close()
    return total_cer / eval_stop_iter



def calc_wer(seqs_hat, seqs_true):
    word_eds, word_ref_lens = [], []
    for i, seq_hat_text in enumerate(seqs_hat):
        seq_true_text = seqs_true[i]
        hyp_words = seq_hat_text.split()
        ref_words = seq_true_text.split()
        # print('hyp_words:', hyp_words, 'ref_words', ref_words)
        word_eds.append(editdistance.eval(hyp_words, ref_words))
        # print('word_eds', word_eds)
        word_ref_lens.append(len(ref_words))
        # print('len(ref_words)', len(ref_words))
    return float(sum(word_eds)) / sum(word_ref_lens)


if __name__=="__main__":
    tr_set, dv_set, feat_dim, vocab_size, tokenizer, msg = \
            load_dataset(n_jobs = 8, use_gpu=True, pin_memory = True,
                         ascending=False, **config['data'])
    model = MyTransformer(sound_dim=120, text_dim=38, d_model=256, dim_feedforward=1024, dropout_rate=0.0, device=device)
    model = model.to(device)
    # model = nn.DataParallel(model)
    model.load_state_dict(torch.load('./ckpt/dev_rus/Weight_BEST.pth'))
    # model.eval()

    tokinazer = CharacterTextEncoder.load_from_file(config['data']['text']['vocab_file'])
    # it = iter(dv_set)
    max_len = 50
    batch_size = config['data']['corpus']['batch_size']
    
    total_cer =  evaluate(model=model, dv_set=dv_set, eval_stop_iter=10000, device=device, config=config, write_txt=True)
    print(total_cer)
    print('Done')