import argparse
import yaml
from dataset import load_dataset
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import numpy as np

from model.model import MyTransformer
from model.cer import Evaluater
from model.metrics import LabelSmoothingLoss

config = yaml.load(open("asr_russian.yaml", 'r'))

def fetch_data(data, device):
	''' Move data to device and compute text seq. length'''
	_, feat, feat_len, txt_in, txt_out = data
	# torch.save(data, "data.pt")
	# input()
	feat = feat.to(device)
	feat_len = feat_len.to(device)
	txt_in = txt_in.to(device)
	txt_out = txt_out.to(device)
	txt_len = torch.sum(txt_out != 0, dim=-1)

	return feat, feat_len, txt_in, txt_out, txt_len

def smooth(step):
	eps_min = 0.1
	eps_max = 0.8
	delay = 180000
	return eps_min + eps_max * np.exp(- step / delay)


parser = argparse.ArgumentParser()
parser.add_argument("--rank", default=0)
parser.add_argument("--world_size", default=4)
parser.add_argument("--init_file", default="/storage/FinogeevE/prj/SpeechRecognition/Finogeev_bucket/init_file")
parser.add_argument("--smoothing", default=0.1, type=float)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--momentum", default=0.95, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--sound_dim", default=120, type=int)
parser.add_argument("--text_dim", default=38, type=int)
parser.add_argument("--d_model", default=256, type=int)
parser.add_argument("--dim_feedforward", default=1024, type=int)
parser.add_argument("--dropout_rate", default=0.0, type=float)
parser.add_argument("--eval_freq", default=2500, type=int)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--eps", default=1e-9, type=float)
parser.add_argument("--k", default=10, type=int)
parser.add_argument("--warmup", default=25000, type=int)

args = parser.parse_args()



if __name__=="__main__":
	print('Start')
	# dist.init_process_group("nccl", init_method=f"file://{args.init_file}", world_size=args.world_size, rank=int(args.rank))

	tr_set, dv_set, feat_dim, vocab_size, tokenizer, msg = \
			load_dataset(n_jobs = 3, use_gpu=True, pin_memory = True,
						 ascending=False, **config['data'])
	print('Data Loaded')
	device = 'cuda'

	print('Size of set', len(tr_set))
	input()

	iter_num = int(1e6)
	model = MyTransformer(sound_dim=args.sound_dim, text_dim=args.text_dim, d_model=args.d_model, dim_feedforward=args.dim_feedforward, 
		dropout_rate=args.dropout_rate, device=device)
	model = model.to(device)
	# model = torch.nn.parallel.DistributedDataParallel(model)
	print('Connect')
	model = nn.DataParallel(model)
	model.load_state_dict(torch.load('./ckpt/dev_weights/Weight_299859.pth'))

	criterion = LabelSmoothingLoss(38, smoothing=args.smoothing, padding_idx=0, normalize_length=True,
				 criterion=nn.KLDivLoss(reduction='none'))
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)
	
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50000, 80000, 100000, 130000, 160000, 190000], gamma=0.15)

	evaluater = Evaluater(config['data']['text']['vocab_file'])
	logger = SummaryWriter()
	best_cer = float('inf')
	it_train = iter(tr_set)
