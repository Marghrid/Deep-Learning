from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable
import time


class Solver(object):
	default_adam_args = {"lr": 1e-4,
						 "betas": (0.9, 0.999),
						 "eps": 1e-8,
						 "weight_decay": 0.0}

	def __init__(self, optim=torch.optim.Adam, optim_args={},
				 loss_func=torch.nn.CrossEntropyLoss()):
		optim_args_merged = self.default_adam_args.copy()
		optim_args_merged.update(optim_args)
		self.optim_args = optim_args_merged
		self.optim = optim
		self.loss_func = loss_func

		self._reset_histories()

	def _reset_histories(self):
		"""
		Resets train and val histories for the accuracy and the loss.
		"""
		self.train_loss_history = []
		self.train_acc_history = []
		self.val_acc_history = []
		self.val_loss_history = []

	def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
		"""
		Train a given model with the provided data.

		Inputs:
		- model: model object initialized from a torch.nn.Module
		- train_loader: train data in torch.utils.data.DataLoader
		- val_loader: val data in torch.utils.data.DataLoader
		- num_epochs: total number of training epochs
		- log_nth: log training accuracy and loss every nth iteration
		"""
		optim = self.optim(model.parameters(), **self.optim_args)
		self._reset_histories()
		iter_per_epoch = len(train_loader)
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		model.to(device)

		print('START TRAIN.')
		start_time = time.time()
		########################################################################
		# TODO:                                                                #
		# Write your own personal training method for our solver.In each      #
		# epoch iter_per_epoch shuffled training batches are processed. The    #
		# loss for each batch is stored in self.train_loss_history. Every      #
		# log_nth iteration the loss is logged. After one epoch the training   #
		# accuracy of the last mini batch is logged and stored in              #
		# self.train_acc_history. We validate at the end of each epoch, log    #
		# the result and store the accuracy of the entire validation set in    #
		# self.val_acc_history.                                                #
		#                                                                      #
		# Your logging could like something like:                              #
		#   ...                                                                #
		#   [Iteration 700/4800] TRAIN loss: 1.452                             #
		#   [Iteration 800/4800] TRAIN loss: 1.409                             #
		#   [Iteration 900/4800] TRAIN loss: 1.374                             #
		#   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
		#   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
		#   ...                                                                #
		########################################################################

		num_iterations = num_epochs * iter_per_epoch
		it = 0

		for epoch in range(num_epochs):
			model.train()
			#train_loader_it = iter(train_loader)

			# In each epoch iter_per_epoch shuffled training batches are processed.
			train_scores = []
			total = 0

			for batch in train_loader:

				#batch = next(train_loader_it)
				it+=1

				x = batch[0].to(device)
				labels = batch[1].to(device)

				optim.zero_grad()   # zero the gradient buffers

				output = model(x)
				loss = self.loss_func(output, labels)

				loss.backward()
				optim.step()

				_, predicted = torch.max(output, 1)
				labels_mask = labels >= 0
				total += labels.size(0)
				train_scores.append(np.mean((predicted == labels)[labels_mask].data.cpu().numpy()))
					

				# Every log_nth iteration the loss is logged.
				if it % log_nth == 0:    # print every log_nth mini-batches
					print('[Iteration %5d, %d] TRAIN loss: %.3f ;  TIME: %d min' %
						(it, num_iterations, loss.data.item(), (time.time() - start_time)//60))
					
			# The loss for each batch is stored in self.train_loss_history.
			#  Makes no sense to save for each batch! Then I have more training
			#  losses than validation losses
			self.train_loss_history.append(loss.data.item())
			# After one epoch the training accuracy of the last mini batch is logged
			#  and stored in self.train_acc_history.
			train_acc = np.mean(train_scores)
			self.train_acc_history.append(train_acc)

			# Format: [Epoch 1/5] TRAIN acc/loss: 0.560/1.374
			print("[Epoch {}/{}] TRAIN acc/loss: {:.3f}/{:.3f}".format(epoch+1, num_epochs, 
				self.train_acc_history[-1], self.train_loss_history[-1]))
			#self.save(args.ckpt_dir, args.ckpt_name, epoch+1)

			# We validate at the end of each epoch, log the result and store the accuracy
			#  of the entire validation set in self.val_acc_history.
			model.eval()
			val_scores = []
			total = 0
			with torch.no_grad():
				for batch in val_loader:
					x, labels = batch

					output = model(x)
					loss = self.loss_func(output, labels)

					_, predicted = torch.max(output, 1)
					labels_mask = labels >= 0
					val_scores.append(np.mean((predicted == labels)[labels_mask].data.cpu().numpy()))

			self.val_loss_history.append(loss.data.item())
			val_acc = np.mean(val_scores)
			self.val_acc_history.append(val_acc)

			print("[Epoch {}/{}] VAL   acc/loss: {:.3f}/{:.3f}".format(epoch+1, num_epochs, 
				self.val_acc_history[-1], self.val_loss_history[-1]))

		########################################################################
		#                             END OF YOUR CODE                         #
		########################################################################
		print('FINISH.')
