import torchvision.transforms as tf
import torch
import torch.nn as nn

from utils import find_anchor_means, gather_outputs

import metrics
import numpy as np


def eval_osr(train_loader, testloader, net, device, config, wandb=None):
	all_accuracy = []


	net = net.to(device)
	net.eval()
	

	#find mean anchors for each class
	anchor_means = find_anchor_means(train_loader, net, device, config, only_correct = True) # finds the training set anchors
	net.set_anchors(torch.Tensor(anchor_means)) # set new anchores as the corrected by the training set

	
	print(f'==> Evaluating open set network accuracy for trial {config.trial}..')
	x, y = gather_outputs(net, testloader, device, data_idx = 1, calculate_scores = True)

	accuracy = metrics.accuracy(x, y)
	all_accuracy += [accuracy]

	# print(f'==> Evaluating open set network AUROC for trial {config.trial}...')
	# xK, yK = gather_outputs(net, testloader, data_idx = 1, calculate_scores = True)
	# xU, yU = gather_outputs(net, testloader, data_idx = 1, calculate_scores = True, unknown = True)

	# auroc = metrics.auroc(xK, xU)
	# all_auroc += [auroc]

	# mean_auroc = np.mean(all_auroc)
	mean_acc = np.mean(all_accuracy)

	print('Raw Top-1 Accuracy: {}'.format(all_accuracy))
	# print('Raw AUROC: {}'.format(all_auroc))
	print('Average Top-1 Accuracy: {}'.format(mean_acc))
	# print('Average AUROC: {}'.format(mean_auroc))