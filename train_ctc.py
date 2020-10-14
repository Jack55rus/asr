import argparse
import yaml
from dataset import load_dataset
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import numpy as np

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

	return feat, feat_len, txt_in, txt_out, txt_len, is_match

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
parser.add_argument("--lambda_coef", default=0.3, type=float)

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

	print('Size of train set', len(tr_set))
	print('Size of dev set', len(dv_set))

	iter_num = int(5e5)
	model = MyTransformer(sound_dim=args.sound_dim, text_dim=args.text_dim, d_model=args.d_model, dim_feedforward=args.dim_feedforward, 
		dropout_rate=args.dropout_rate, device=device)
	model = model.to(device)
	# model = torch.nn.parallel.DistributedDataParallel(model)
	print('Connect')
	# model = nn.DataParallel(model)
	# model.load_state_dict(torch.load('./ckpt/Weight_BEST.pth'))

	criterion_ce = LabelSmoothingLoss(38, smoothing=args.smoothing, padding_idx=0, normalize_length=True,
				 criterion=nn.KLDivLoss(reduction='none'))
	criterion_ctc = nn.CTCLoss(blank=38, zero_infinity=True)
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
	print('Number of iters per epoch:', len(tr_set)/config['data']['corpus']['batch_size'])
	for iteration in range(66001, iter_num):
		# print('iter:', iteration)
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
		out_ce, out_ctc, new_feat_len = model(feat, txt_in)
		# print(out_ce.shape, out_ctc.shape, new_feat_len.shape)
		# print(txt_out.shape, new_feat_len.shape, txt_len)

		loss_ctc = criterion_ctc(out_ctc.permute(1, 0, 2).log_softmax(2), txt_out, new_feat_len, txt_len)
		# loss_ce = criterion_ce(out_ce.view(-1, 31), txt_out.view(-1))
		loss_ce = criterion_ce(out_ce, txt_out)
		loss = (1 - args.lambda_coef)*loss_ctc + args.lambda_coef*loss_ce
		loss.backward()
		optimizer.step()

		if iteration % 10 == 0:
			# logger.add_scalar('Loss', {'Loss_ce/train': loss_ce.item(), 'Loss_ctc/train': loss_ctc.item(), 'Total Loss/train': loss.item()}, iteration)
			logger.add_scalar('Loss/train', loss.item(), iteration)
			logger.add_scalar('Loss_ce/train', loss_ce.item(), iteration)
			logger.add_scalar('Loss_ctc/train', loss_ctc.item(), iteration)
			print('ce', loss_ce.item(), 'ctc', loss_ctc.item())
			
		if iteration % args.eval_freq == 0:
			# result = evaluater.evaluate(out_ce.argmax(dim=-1), txt_out, visual=True)
			max_eval_iter = 50
			total_cer = evaluate(model=model, dv_set=dv_set, eval_stop_iter=max_eval_iter, device=device, config=config)
			if total_cer < best_cer:
				print('Saving results with new best cer')
				best_cer = total_cer
				torch.save(model.state_dict(), './ckpt/Weight_'+'BEST'+'.pth')
			print('ce', loss_ce.item(), 'ctc', loss_ctc.item())
			print('iteration:', iteration, 'total cer:', total_cer)
			logger.add_scalar('CER', total_cer, iteration)
			# for param_group in optimizer.param_groups:
			# 	logger.add_scalar('Loss/lr', param_group['lr'], iteration)
			# 	break
			# model = nn.DataParallel(model)
		total_result = 0
		if iteration % args.eval_freq == 0:
			print('Saving weights...')
			torch.save(model.state_dict(), './ckpt/Weight_'+str(iteration)+'.pth')
		
	
