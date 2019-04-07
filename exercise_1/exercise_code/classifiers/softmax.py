"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np
import math

from .linear_classifier import LinearClassifier

def softmax(f_vector, idx):
	softmax_denominator = np.sum(np.exp(f_vector))
	return np.exp(f_vector[idx]) / softmax_denominator

def cross_entropoy_loss_naive(W, X, y, reg):
	"""
	Cross-entropy loss function, naive implementation (with loops)

	Inputs have dimension D, there are C classes, and we operate on minibatches
	of N examples.

	Inputs:
	- W: A numpy array of shape (D, C) containing weights.
	- X: A numpy array of shape (N, D) containing a minibatch of data.
	- y: A numpy array of shape (N,) containing training labels; y[i] = c means
	  that X[i] has label c, where 0 <= c < C.
	- reg: (float) regularization strength

	Returns a tuple of:
	- loss as single float
	- gradient with respect to weights W; an array of same shape as W
	"""
	# pylint: disable=too-many-locals
	# Initialize the loss and gradient to zero.
	loss = 0.0
	dW = np.zeros_like(W)

	############################################################################
	# TODO: Compute the cross-entropy loss and its gradient using explicit     #
	# loops. Store the loss in loss and the gradient in dW. If you are not     #
	# careful here, it is easy to run into numeric instability. Don't forget   #
	# the regularization!                                                      #
	############################################################################

	N = X.shape[0]
	C = W.shape[1]
	for i in range(N):

		f_i = X[i].dot(W)   # scores for one data point. Shape (C,)
		f_i -= np.max(f_i)  # shift the values of f so that the highest number is 0
							# to avoid numerical instability

		loss -= math.log(softmax(f_i, y[i]))

		for k in range(C):
			p_k = softmax(f_i, k)
			dW[:, k] += (p_k - (k == y[i])) * X[i] #column

	# regularize and normalize
	loss /= N
	loss += 0.5 * reg * np.sum(W.T.dot(W))

	dW /= N
	dW += 0.5 * reg * W


	############################################################################
	#                          END OF YOUR CODE                                #
	############################################################################

	return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
	"""
	Cross-entropy loss function, vectorized version.

	Inputs and outputs are the same as in cross_entropoy_loss_naive.
	"""
	# Initialize the loss and gradient to zero.
	loss = 0.0
	dW = np.zeros_like(W)

	############################################################################
	# TODO: Compute the cross-entropy loss and its gradient without explicit   #
	# loops. Store the loss in loss and the gradient in dW. If you are not     #
	# careful here, it is easy to run into numeric instability. Don't forget   #
	# the regularization!                                                      #
	############################################################################

	N = X.shape[0]

	f = X.dot(W)  # Matrix with scores (N, C)
	f -= np.max(f, axis=1, keepdims=True)  # shift the values of f so that the highest number is 0
							               # to avoid numerical instability

	sum_f = np.sum(np.exp(f), axis=1, keepdims=True) # softmax denominators. Shape (C,)
	p = np.exp(f)/sum_f  #matrix with result of softmax

	loss = np.sum(-np.log(p[np.arange(N), y]))

	one_hot_y = np.zeros_like(p) # [0, ..., 1, ..., 0]
	one_hot_y[np.arange(N), y] = 1
	dW = X.T.dot(p - one_hot_y)

	# regularize and normalize
	loss /= N
	loss += 0.5 * reg * np.sum(W * W)
	dW /= N
	dW += 0.5 * reg * W

	############################################################################
	#                          END OF YOUR CODE                                #
	############################################################################

	return loss, dW


class SoftmaxClassifier(LinearClassifier):
	"""The softmax classifier which uses the cross-entropy loss."""

	def loss(self, X_batch, y_batch, reg):
		return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
	# results is dictionary mapping tuples of the form
	# (learning_rate, regularization_strength) to tuples of the form
	# (training_accuracy, validation_accuracy). The accuracy is simply the
	# fraction of data points that are correctly classified.
	results = {}
	best_val = -1
	best_softmax = None
	all_classifiers = []
	learning_rates = [1e-7, 5e-7, 1e-6, 5e-6]
	regularization_strengths = [1e2, 2.5e2, 5e2, 7.5e2, 1e3, 2.5e3, 5e3, 7.5e3]

	############################################################################
	# TODO:                                                                    #
	# Write code that chooses the best hyperparameters by tuning on the        #
	# validation set. For each combination of hyperparameters, train a         #
	# classifier on the training set, compute its accuracy on the training and #
	# validation sets, and  store these numbers in the results dictionary.     #
	# In addition, store the best validation accuracy in best_val and the      #
	# Softmax object that achieves this accuracy in best_softmax.              #                                      #
	#                                                                          #
	# Hint: You should use a small value for num_iters as you develop your     #
	# validation code so that the classifiers don't take much time to train;   # 
	# once you are confident that your validation code works, you should rerun #
	# the validation code with a larger value for num_iters.                   #
	############################################################################

	num_iters = 5000

	for learning_rate in learning_rates:
		for regularization_strength in regularization_strengths:
			softmax = SoftmaxClassifier()
			softmax.train(X_train, y_train, 
				learning_rate=learning_rate,
				reg=regularization_strength, 
				num_iters=num_iters, verbose=False)

			y_train_pred = softmax.predict(X_train)
			training_accuracy = np.mean(y_train == y_train_pred)
			y_val_pred = softmax.predict(X_val)
			validation_accuracy = np.mean(y_val == y_val_pred)

			results[(learning_rate, regularization_strength)] = (training_accuracy, validation_accuracy)

			if(validation_accuracy > best_val):
				best_val = validation_accuracy
				best_softmax = softmax

			all_classifiers.append((softmax, validation_accuracy))




	############################################################################
	#                              END OF YOUR CODE                            #
	############################################################################

	# Print out results.
	for (lr, reg) in sorted(results):
		train_accuracy, val_accuracy = results[(lr, reg)]
		print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
			lr, reg, train_accuracy, val_accuracy))
		
		print('best validation accuracy achieved during validation: %f' % best_val)

	return best_softmax, results, all_classifiers
