import numpy as np


def conv_forward_naive(x, w, b, conv_param):
		"""
		A naive implementation of the forward pass for a convolutional layer.

		The input consists of N data points, each with C channels, height H and width
		W. We convolve each input with F different filters, where each filter spans
		all C channels and has height HH and width WW.

		Input:
		- x: Input data of shape (N, C, H, W)
		- w: Filter weights of shape (F, C, HH, WW)
		- b: Biases, of shape (F,)
		- conv_param: A dictionary with the following keys:
			- 'stride': The number of pixels between adjacent receptive fields in the
				horizontal and vertical directions.
			- 'pad': The number of pixels that will be used to zero-pad the input.

		Returns a tuple of:
		- out: Output data, of shape (N, F, H', W') where H' and W' are given by
			H' = 1 + (H + 2 * pad - HH) / stride
			W' = 1 + (W + 2 * pad - WW) / stride
		- cache: (x, w, b, conv_param) for the backward pass
		"""
		out = None
		#############################################################################
		# TODO: Implement the convolutional forward pass.                           #
		# Hint: you can use the function np.pad for padding.                        #
		#############################################################################
		pad = conv_param['pad']
		stride = conv_param['stride']

		F, C, HH, WW = w.shape
		N, C, H, W = x.shape
		Hp = 1 + (H + 2 * pad - HH) // stride
		Wp = 1 + (W + 2 * pad - WW) // stride

		out = np.zeros((N, F, Hp, Wp))

		padded = np.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant', constant_values=0)
		
		for i in range(N): # ith example
			for j in range(F): # jth filter
	
				# Convolve this filter over windows
				for k in range(Hp):
					hs = k * stride
					for l in range(Wp):
						ws = l * stride
	
						# Window we want to apply the respective jth filter over (C, HH, WW)
						window = padded[i, :, hs:hs+HH, ws:ws+WW]
	
						# Convolve
						out[i, j, k, l] = np.sum(window*w[j]) + b[j]
	
		
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
		cache = (x, w, b, conv_param)
		return out, cache


def conv_backward_naive(dout, cache):
		"""
		A naive implementation of the backward pass for a convolutional layer.

		Inputs:
		- dout: Upstream derivatives.
		- cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

		Returns a tuple of:
		- dx: Gradient with respect to x
		- dw: Gradient with respect to w
		- db: Gradient with respect to b
		"""
		dx, dw, db = None, None, None
		#############################################################################
		# TODO: Implement the convolutional backward pass.                          #
		#############################################################################
		x, w, b, conv_param = cache
		pad = conv_param['pad']
		stride = conv_param['stride']
		F, C, HH, WW = w.shape
		N, C, H, W = x.shape
		Hp = 1 + (H + 2 * pad - HH) // stride
		Wp = 1 + (W + 2 * pad - WW) // stride
	
		dx = np.zeros_like(x)
		dw = np.zeros_like(w)
		db = np.zeros_like(b)
	
		# Add padding around each 2D image (and respective gradient)
		# Contribute to the boundary sums and in some cases the
		# only way to do that is by writing into the padding.
		padded = np.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
		padded_dx = np.pad(dx, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
	
		for i in range(N): # ith example
			for j in range(F): # jth filter
				# Convolve this filter over windows
				for k in range(Hp):
					hs = k * stride
					for l in range(Wp):
						ws = l * stride
	
						# Window we applies the respective jth filter over (C, HH, WW)
						window = padded[i, :, hs:hs+HH, ws:ws+WW]
	
						# Compute gradient of out[i, j, k, l] = np.sum(window*w[j]) + b[j]
						db[j] += dout[i, j, k, l]
						dw[j] += window*dout[i, j, k, l]
						padded_dx[i, :, hs:hs+HH, ws:ws+WW] += w[j] * dout[i, j, k, l]
	
		# "Unpad"
		dx = padded_dx[:, :, pad:pad+H, pad:pad+W]
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
		return dx, dw, db


def max_pool_forward_naive(x, pool_param):
		"""
		A naive implementation of the forward pass for a max pooling layer.

		Inputs:
		- x: Input data, of shape (N, C, H, W)
		- pool_param: dictionary with the following keys:
			- 'pool_height': The height of each pooling region
			- 'pool_width': The width of each pooling region
			- 'stride': The distance between adjacent pooling regions

		Returns a tuple of:
		- out: Output data
		- cache: (x, maxIdx, pool_param) for the backward pass with maxIdx, of shape (N, C, H, W, 2)
		"""
		out = None
		#############################################################################
		# TODO: Implement the max pooling forward pass                              #
		#############################################################################
		HH = pool_param['pool_height']
		WW = pool_param['pool_width']
		stride = pool_param['stride']
		N, C, H, W = x.shape
		Hp = 1 + (H - HH) // stride
		Wp = 1 + (W - WW) // stride
		out = np.zeros((N, C, Hp, Wp))
		
 
		for i in range(N):
		# Need this; apparently we are required to max separately over each channel
			for j in range(C):
				for k in range(Hp):
					hs = k * stride
					for l in range(Wp):
						ws = l * stride
					
						# Window (C, HH, WW)
						window = x[i, j, hs:hs+HH, ws:ws+WW]
						out[i, j, k, l] = np.max(window)
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
		cache = (x, pool_param)
		return out, cache


def max_pool_backward_naive(dout, cache):
		"""
		A naive implementation of the backward pass for a max pooling layer.

		Inputs:
		- dout: Upstream derivatives
		- cache: A tuple of (x, pool_param) as in the forward pass.

		Returns:
		- dx: Gradient with respect to x
		"""
		dx = None
		#############################################################################
		# TODO: Implement the max pooling backward pass                             #
		#############################################################################
		x, pool_param = cache
		HH = pool_param['pool_height']
		WW = pool_param['pool_width']
		stride = pool_param['stride']
		N, C, H, W = x.shape
		Hp = 1 + (H - HH) // stride
		Wp = 1 + (W - WW) // stride

		dx = np.zeros_like(x)

		for i in range(N):
			for j in range(C):
				for k in range(Hp):
					hs = k * stride
					for l in range(Wp):
						ws = l * stride

						# Window (C, HH, WW)
						window = x[i, j, hs:hs+HH, ws:ws+WW]
						m = np.max(window)

						# Gradient of max is indicator
						dx[i, j, hs:hs+HH, ws:ws+WW] += (window == m) * dout[i, j, k, l]

		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
		return dx
		