"""
	Train an open set classifier with CAC Loss on the datasets.

	The overall setup of this training script has been adapted from https://github.com/kuangliu/pytorch-cifar

	Dimity Miller, 2020
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tf
from utils import progress_bar
import os
import numpy as np
import copy



def CACLoss(distances, gt, lbda_anchor_loss, known_classes):
	'''Returns CAC loss, as well as the Anchor and Tuplet loss components separately for visualisation.'''
	true = torch.gather(distances, 1, gt.view(-1, 1)).view(-1)
	non_gt = torch.Tensor([[i for i in range(known_classes) if gt[x] != i] for x in range(len(distances))]).long().cuda()
	others = torch.gather(distances, 1, non_gt)
	
	anchor = torch.mean(true)

	tuplet = torch.exp(-others+true.unsqueeze(1))
	tuplet = torch.mean(torch.log(1+torch.sum(tuplet, dim = 1)))

	total = lbda_anchor_loss*anchor + tuplet

	return total, anchor, tuplet


# Training
def train(trainloader, epoch, optimizer, net, device, config, wandb=None):
	print('\nEpoch: %d' % epoch)
	if config.use_wandb_flag and wandb is not None:
		print('\nEpoch inside: %d' % epoch)
		wandb.log({"epoch": int(epoch)}, step=(epoch-1)*len(trainloader))
	net.train()
	train_loss = 0
	correctDist = 0
	total = 0


	for batch_idx, (inputs, targets) in enumerate(trainloader):
		inputs, targets = inputs.to(device), targets.to(device)
		#convert from original dataset label to known class label

		optimizer.zero_grad()

		outputs = net(inputs)
		cacLoss, anchorLoss, tupletLoss = CACLoss(outputs[1], targets, config.lbda_anchor_loss, config.num_classes)

		if config.use_wandb_flag and  batch_idx%3 == 0 and wandb is not None:
			wandb.log({"train/CAC_Loss": cacLoss.item(), 
	      'train/anchor_Loss': anchorLoss.item(),
		    'train/tuplet_Loss': tupletLoss.item()}
			, step=batch_idx + (epoch-1)*len(trainloader))

		cacLoss.backward()

		optimizer.step()

		train_loss += cacLoss.item()

		_, predicted = outputs[1].min(1)

		total += targets.size(0)
		correctDist += predicted.eq(targets).sum().item()

		progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correctDist/total, correctDist, total))
	if config.use_wandb_flag and wandb is not None:
		acc = 100.*correctDist/total
		wandb.log({'train/accuracy': acc}, step=epoch)

def val(valloader, epoch, net, device, config, best_acc, best_anchor, best_cac, best_net, wandb=None):
	net.eval()
	anchor_loss = 0
	cac_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(valloader):
			inputs = inputs.to(device)
			targets = targets.to(device)

			outputs = net(inputs)

			cacLoss, anchorLoss, tupletLoss = CACLoss(outputs[1], targets, config.lbda_anchor_loss, config.num_classes)

			anchor_loss += anchorLoss
			cac_loss += cacLoss

			_, predicted = outputs[1].min(1)
			
			total += targets.size(0)

			correct += predicted.eq(targets).sum().item()

			progress_bar(batch_idx, len(valloader), 'Acc: %.3f%% (%d/%d)'
				% (100.*correct/total, correct, total))
   
	anchor_loss /= len(valloader)
	cac_loss /= len(valloader)
	acc = 100.*correct/total

	# Save checkpoint.
	state = {
		'net': net.state_dict(),
		'acc': acc,
		'epoch': epoch,
	}
	if not os.path.isdir('networks/weights'):
		os.mkdir('networks/weights')
	if not os.path.isdir('networks/weights/{}'.format(config.dataset)):
		os.mkdir('networks/weights/{}'.format(config.dataset))
	save_name = '{}_CACclassifier_'.format(config.trial)
	if anchor_loss <= best_anchor:
		print('Saving..')
		torch.save(state, 'networks/weights/{}/'.format(config.dataset)+save_name+'AnchorLoss.pth')
		best_anchor = anchor_loss
		best_net = copy.deepcopy(net.state_dict())

	if cac_loss <= best_cac:
		print('Saving..')
		torch.save(state, 'networks/weights/{}/'.format(config.dataset)+save_name+'CACLoss.pth')
		best_cac = cac_loss

	if acc >= best_acc:
		print('Saving..')
		torch.save(state, 'networks/weights/{}/'.format(config.dataset)+save_name+'Accuracy.pth')
		best_acc = acc

	
	if config.use_wandb_flag and wandb is not None:
		print(f'in val wandb logger, epoch {epoch}')
		val_log= {"val/accuracy": acc, 
	      'val/anchorLoss': float(anchorLoss.detach().cpu().float()),
		    'val/CACLoss': float(cac_loss.detach().cpu().float())}
		print(val_log)
		wandb.log(val_log
			, step=epoch)
	return best_acc, best_anchor, best_cac, best_net
	


def train_osr(trainloader, valloader, net, device, config, optimizer, wandb=None):
	#parameters useful when resuming and finetuning
	best_acc = 0
	best_cac = 10000
	best_anchor = 10000
	start_epoch = 1
	best_net = None
	print('==> Building Anchores..')
	# initialising with anchors
	anchors = torch.diag(torch.Tensor([config.alpha_magnitude for i in range(config.num_classes)]))	
	net.set_anchors(anchors)

	net = net.to(device)
	training_iter = int(config.resume)

	if config.resume:
		# Load checkpoint.
		print('==> Resuming from checkpoint..')
		assert os.path.isdir('networks/weights'), 'Error: no checkpoint directory found!'
		
		checkpoint = torch.load('networks/weights/{}/{}_CACclassifier_AnchorLoss.pth'.format(config.dataset, config.trial))

		start_epoch = checkpoint['epoch']

		net_dict = net.state_dict()
		pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in net_dict}
		net.load_state_dict(pretrained_dict)

	net.train()
	


	max_epoch = config.epochs
	for epoch in range(start_epoch, max_epoch + 1):
		train(trainloader, epoch, optimizer, net, device, config, wandb)
		best_acc, best_anchor, best_cac, best_net = val(valloader, epoch, net, device, config, best_acc, best_anchor, best_cac, best_net, wandb)
	return best_net









