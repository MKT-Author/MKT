"""
basic trainer
"""
import os
import time

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils as utils
import numpy as np
import torch

__all__ = ["Trainer"]


class Trainer(object):
	"""
	trainer for training network, use SGD
	"""
	
	def __init__(self, model, model_teacher, test_loader, settings, ckpt_file):
		"""
		init trainer
		"""
		self.settings = settings
		
		self.model = utils.data_parallel(
			model, self.settings.nGPU, self.settings.GPU)
		self.model_teacher = utils.data_parallel(
			model_teacher, self.settings.nGPU, self.settings.GPU)

		# load weight file and set model parameters
		self.test_loader = test_loader
		state_dict = torch.load(ckpt_file, map_location='cuda')
		self.set_state_dict(state_dict)


	def test(self):
		"""
		testing
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()
		
		self.model.eval()
		self.model_teacher.eval()
		
		iters = len(self.test_loader)
		start_time = time.time()
		end_time = start_time

		for i, (images, labels) in enumerate(self.test_loader):
			start_time = time.time()

			labels = labels.cuda()
			images = images.cuda()
			output = self.model(images)

			loss = torch.ones(1)

			# compare top-1 and top-5 accuracy

			single_error, single_loss, single5_error = utils.compute_singlecrop(
				outputs=output, loss=loss,
				labels=labels, top5_flag=True, mean_flag=True)

			top1_error.update(single_error, images.size(0))
			top1_loss.update(single_loss, images.size(0))
			top5_error.update(single5_error, images.size(0))

			end_time = time.time()

		return top1_error.avg, top1_loss.avg, top5_error.avg


	def set_state_dict(self, state_dict):
		if 'model' in state_dict:
			self.model.load_state_dict(state_dict['model'])

		else:
			new_state_dict = {}
			for k, v in state_dict.items():
				new_state_dict[k[7:]] = v

			self.model.load_state_dict(new_state_dict)
