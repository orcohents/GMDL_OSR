"""
	Network definition for our proposed CAC open set classifier. 

	Dimity Miller, 2020
"""

import torch
import torchvision
import torch.nn as nn

class openSetClassifier(nn.Module):
	def __init__(self, num_classes = 10, num_channels = 1, init_weights = False, dropout = 0.3, **kwargs):
		super(openSetClassifier, self).__init__()

		self.num_classes = num_classes
		self.encoder = BaseEncoder(num_channels, init_weights, dropout)
		
		self.classify = nn.Linear(128*7*7, num_classes)

		self.anchors = nn.Parameter(torch.zeros(self.num_classes, self.num_classes).double(), requires_grad = False)

		if init_weights:
			self._initialize_weights()
		
		self.cuda()


	def forward(self, x, skip_distance = False):
		batch_size = len(x)

		x = self.encoder(x)
		x = x.view(batch_size, -1)

		outLinear = self.classify(x)

		if skip_distance:
			return outLinear, None

		outDistance = self.distance_classifier(outLinear)

		return outLinear, outDistance

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

	def set_anchors(self, means):
		self.anchors = nn.Parameter(means.double(), requires_grad = False)
		self.cuda()

	def distance_classifier(self, x):
		''' Calculates euclidean distance from x to each class anchor
			Returns n x m array of distance from input of batch_size n to anchors of size m
		'''

		n = x.size(0)
		m = self.num_classes
		d = self.num_classes

		x = x.unsqueeze(1).expand(n, m, d).double()
		anchors = self.anchors.unsqueeze(0).expand(n, m, d)
		dists = torch.norm(x-anchors, 2, 2)

		return dists

class BaseEncoder(nn.Module):
	def __init__(self, num_channels, init_weights, dropout = 0.3, **kwargs): 
		super().__init__()
		self.dropout = nn.Dropout2d(dropout)
		self.relu = nn.LeakyReLU(0.2)
		
		self.bn1 = nn.BatchNorm2d(64)
		self.bn2 = nn.BatchNorm2d(64)
		self.bn3 = nn.BatchNorm2d(128)

		self.bn4 = nn.BatchNorm2d(128)
		self.bn5 = nn.BatchNorm2d(128)
		self.bn6 = nn.BatchNorm2d(128)

		self.bn7 = nn.BatchNorm2d(128)
		self.bn8 = nn.BatchNorm2d(128)
		self.bn9 = nn.BatchNorm2d(128)

		self.conv1 = nn.Conv2d(num_channels,       64,     3, 1, 1, bias=False)
		self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
		self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

		self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
		self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
		self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

		self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
		self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)


		self.encoder1 = nn.Sequential(
		                    self.conv1,
		                    self.bn1,
		                    self.relu,
		                    self.conv2,
		                    self.bn2,
		                    self.relu,
		                    self.conv3,
		                    self.bn3,
		                    self.relu,
		                    self.dropout,
		                )

		self.encoder2 = nn.Sequential(
		                        self.conv4,
		                        self.bn4,
		                        self.relu,
		                        self.conv5,
		                        self.bn5,
		                        self.relu,
		                        self.conv6,
		                        self.bn6,
		                        self.relu,
		                        self.dropout,
		                    )

		self.encoder3 = nn.Sequential(
		                        self.conv7,
		                        self.bn7,
		                        self.relu,
		                        self.conv8,
		                        self.bn8,
		                        self.relu,
		                        self.dropout,
								
		                    )

		if init_weights:
			self._initialize_weights()
	
		self.cuda()


	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x1 = self.encoder1(x)
		x2 = self.encoder2(x1)
		x3 = self.encoder3(x2)
		return x3




