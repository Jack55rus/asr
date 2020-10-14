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

def adjust_lr(optimizer, step):
	lr = args.k * (args.d_model ** (-0.5)) * min(step ** (-0.5), step * (args.warmup ** (-1.5)))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


if __name__=="__main__":
	print('Start')
	# dist.init_process_group("nccl", init_method=f"file://{args.init_file}", world_size=args.world_size, rank=int(args.rank))

	tr_set, dv_set, feat_dim, vocab_size, tokenizer, msg = \
			load_dataset(n_jobs = 3, use_gpu=True, pin_memory = True,
						 ascending=False, **config['data'])
	print('Data Loaded')
	device = 'cuda'

	print('Size of set', len(tr_set))

	iter_num = int(1e5)
	model = MyTransformer(sound_dim=args.sound_dim, text_dim=args.text_dim, d_model=args.d_model, dim_feedforward=args.dim_feedforward, 
		dropout_rate=args.dropout_rate, device=device)
	model = model.to(device)
	# model = torch.nn.parallel.DistributedDataParallel(model)
	print('Connect')
	model = nn.DataParallel(model)
	# model.load_state_dict(torch.load('./ckpt/dev_weights/Weight_299859.pth'))

	criterion = LabelSmoothingLoss(38, smoothing=args.smoothing, padding_idx=0, normalize_length=True,
				 criterion=nn.KLDivLoss(reduction='none'))
	# criterion_ctc = nn.CTCLoss()
	# criterion_ce = nn.CrossEntropyLoss(ignore_index=0)
	# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)
	# optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-06, weight_decay=0)
	
	# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16000, 32000, 64000, 80000, 160000, 190000], gamma=0.15)

	evaluater = Evaluater(config['data']['text']['vocab_file'])
	logger = SummaryWriter()
	best_cer = float('inf')
	it_train = iter(tr_set)
	print('Start learning')
	print('Number of iters per epoch:', len(tr_set))
	for iteration in range(1, iter_num):
		# adjust_lr(optimizer, iteration)
		# print(iteration)
		try:
			batch = next(it_train)
		except:
			it_train = iter(tr_set)
			batch = next(it_train)

		feat, feat_len, txt_in, txt_out, txt_len = fetch_data(batch, device)
		# print(txt_in, txt_out, sep='\n')
		# input()
		# print('feat_len', feat_len.shape)
		# input('stop')

		optimizer.zero_grad()

		out = model(feat, txt_in)

		loss = criterion(out, txt_out)
		# print('feat_len before', feat_len)
		# print('out_ctc.log_softmax(2), txt_out, feat_len, txt_len', out_ctc.log_softmax(2).shape, txt_out.shape, feat_len.shape, txt_len.shape)
		# feat_len = feat_len // 4
		# feat_len = feat_len - 1
		# print('feat_len after', feat_len)
		# feat_len = torch.trunc(feat_len)
		# loss_ctc = criterion_ctc(out_ctc.log_softmax(2), txt_out, feat_len, txt_len)
		# loss_ce = criterion_ce(out_ce.view(-1, 37), txt_out.view(-1))
		# print('ctc:', loss_ctc.item(), 'ce:', loss_ce.item())
		# input('stop')
		# loss = loss_ctc + loss_ce
		loss.backward()
		optimizer.step()
		# scheduler.step()
			
		if iteration % 50 == 0:
			print('Loss: ', loss.item())
			# smoothing = smooth(iteration)
			# criterion.smoothing = smoothing
			# criterion.confidence = 1.0 - smoothing
			# print('criterion.smoothing', criterion.smoothing)
			# print('criterion.confidence', criterion.confidence)
			# print('Loss:',loss.item())
			logger.add_scalar('Loss/train', loss.item(), iteration)
			for param_group in optimizer.param_groups:
				logger.add_scalar('Loss/lr', param_group['lr'], iteration)
				break

		total_result = 0
		if iteration % args.eval_freq == 0:
			# model.eval()
			# for i, batch in enumerate(dv_set):
			# 	with torch.no_grad():
			# 		feat, feat_len, txt_in, txt_out, txt_len = fetch_data(batch, device)
			# 		out = model(feat, txt_in)
			# 		if i == 0:
			# 			result, pred_list, true_list = evaluater.evaluate(out.argmax(dim=-1), txt_out, visual_batch=True)
			# 		else:
			# 			result = evaluater.evaluate(out.argmax(dim=-1), txt_out, visual=False)
			# 		total_result += result

			# model.train()
			# for idx,(p,g) in enumerate(zip(pred_list,true_list)):
			# 	logger.add_text('pred_'+str(idx), p, iteration)
			# 	logger.add_text('true_'+str(idx), g, iteration)

			# print('Total error',total_result/len(dv_set))
			# logger.add_scalar('CER/eval', total_result/len(dv_set), iteration)

			# if total_result/len(dv_set) < best_cer:
			# 	best_cer = total_result/len(dv_set)
			# 	print('NEW BEST RESULT', total_result/len(dv_set))
			# 	torch.save(model.state_dict(), './ckpt/Best.pth')

			print('Saving weights...')
			torch.save(model.state_dict(), './ckpt/Weight_'+str(iteration)+'.pth')
		
	
