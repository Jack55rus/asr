import argparse
import yaml
from dataset import load_dataset
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import numpy as np
from copy import deepcopy
from model.model_ctc import MyTransformer
from model.cer import Evaluater
from model.metrics import LabelSmoothingLoss
from inference_ctc import evaluate

config = yaml.load(open("asr_russian.yaml", 'r'))

def fetch_data(data, device):
	''' Move data to device and compute text seq. length'''
	_, feat, feat_len, txt_in, txt_out, is_match = data
	# torch.save(data, "data.pt")
	# input()
	feat = feat.to(device)
	feat_len = feat_len.to(device)
	txt_in = txt_in.to(device)
	txt_out = txt_out.to(device)
	txt_len = torch.sum(txt_out != 0, dim=-1)

	return feat, feat_len, txt_in, txt_out, txt_len, is_match.to(device)


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
parser.add_argument("--lambda_coef", default=0.3, type=float)

args = parser.parse_args()

def eval_bin_acc(model, eval_set, device):
	model = deepcopy(model)
	model.eval()
	correct_results_sum = 0
	total_samples = 0
	with torch.no_grad():
		for i, batch_eval in enumerate(eval_set):
			feat, feat_len, txt_in, txt_out, txt_len, is_match = fetch_data(batch_eval, device)
			y_pred = model(feat, txt_in)
			# print('y_pred', y_pred)
			y_pred_tag = torch.round(torch.sigmoid(y_pred))
			# print('y_pred_tag', y_pred_tag)
			# print('is_match', is_match.unsqueeze(1))
			# input()
			correct_results_sum += (y_pred_tag == is_match.unsqueeze(1)).sum().float()
			# print('correct_results_sum', correct_results_sum)
			total_samples += is_match.shape[0]
			# input()
	return round((correct_results_sum / total_samples).item(), 3)

if __name__=="__main__":
	print('Start')
	# dist.init_process_group("nccl", init_method=f"file://{args.init_file}", world_size=args.world_size, rank=int(args.rank))

	tr_set, dv_set, feat_dim, vocab_size, tokenizer, msg = \
			load_dataset(n_jobs = 3, use_gpu=True, pin_memory = True,
						 ascending=False, **config['data'])
	print('Data Loaded')
	device = 'cuda'

	print('Size of train set', len(tr_set))
	print('Size of dev set', len(dv_set))

	iter_num = int(5e5)
	model = MyTransformer(sound_dim=args.sound_dim, text_dim=args.text_dim, d_model=args.d_model, dim_feedforward=args.dim_feedforward, 
		dropout_rate=args.dropout_rate, device=device, classificator=True)
	model = model.to(device)
	print('Connect')
	model = nn.DataParallel(model)
	model.load_state_dict(torch.load('./ckpt/dev_rus/Weight_BEST.pth'), strict=False)

	criterion = nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)
	
	logger = SummaryWriter()
	best_cer = float('inf')
	it_train = iter(tr_set)
	print('Start learning')
	print('Number of iters per epoch:', len(tr_set)/config['data']['corpus']['batch_size'])
	for iteration in range(1, iter_num):
		try:
			batch = next(it_train)
		except:
			it_train = iter(tr_set)
			batch = next(it_train)

		feat, feat_len, txt_in, txt_out, txt_len, is_match = fetch_data(batch, device)
		# print(feat)
		# print(feat.shape)
		# print(feat_len)
		# print(txt_in)
		# print(txt_in.shape)
		# print(txt_len)
		# print(is_match)
		# input('stop')
		optimizer.zero_grad()
		# print(feat.type(), txt_in.type())
		out_ce = model(feat, txt_in)

		# print(out_ce.shape, out_ctc.shape, new_feat_len.shape)
		# print(txt_out.shape, new_feat_len.shape, txt_len)
		loss = criterion(out_ce, is_match.unsqueeze(1))

		loss.backward()
		optimizer.step()

		if iteration % 10 == 0:
			# print('out_ce', out_ce)
			logger.add_scalar('Loss/train', loss.item(), iteration)
			print('ce', loss.item())
			
		total_result = 0
		if iteration % args.eval_freq == 0:
			acc = eval_bin_acc(model, dv_set, device)
			print('Accuracy', acc)
			model.train()
			print('Saving weights...')
			logger.add_scalar('Accuracy', acc, iteration)
			torch.save(model.state_dict(), './ckpt/Weight_BC_'+str(iteration)+'.pth')
		
	
