"""
This script implements the SCSG optimizer.
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops 
class SCSGOptimizer(tf.train.Optimizer):
	"""
	The optimizer for carrying out SCSG update. 
	"""
	def __init__(self, loss, learning_rate = 1e-3):

		self._learning_rate = learning_rate  

		# Build graph for gradient computation.
		self._grads_and_vars = self._compute_gradients(loss)

		# Build placeholder for variance reduction terms. 
		self._variance_reduction_ph = [tf.placeholder(tf.float32, 
				shape=v.get_shape()) for v in zip(*self._grads_and_vars)[1]]
		self._bias_correction_ph = [tf.placeholder(tf.float32, 
				shape=v.get_shape()) for v in zip(*self._grads_and_vars)[1]]	

		# Build graph for carrying out one update within a batch I_j.
		self._update = self._compute_single_update()	

	def _compute_gradients(self, loss): 
		""" 
		Build graph for gradient computation.
		"""
		opt = tf.train.GradientDescentOptimizer(learning_rate = 1) 
		grads_and_vars = opt.compute_gradients(loss) 
		grads_and_vars = [(g, v) for g, v in grads_and_vars \
		if not isinstance(g, ops.IndexedSlices)]
		grads, tvars = zip(*grads_and_vars) 
		grads_and_vars = zip(grads, tvars)  
		return grads_and_vars


	def _compute_single_update(self):  
		""" 
		Build graph for carrying out one update with a batch I_j.
		"""
		grads_and_vars = self._grads_and_vars

		update_ops = []
		for i, (g, v) in enumerate(grads_and_vars):
			g += self._bias_correction_ph[i] - self._variance_reduction_ph[i]
			
			update = - self._learning_rate * g
			update_ops.append(v.assign_add(update))            

		return tf.group(*update_ops)   

	def batch_update(self,sess,feed_dict, batch_size, mini_batchsize, lr = None): 
		input_ph = feed_dict.keys()
		input_data = feed_dict.values() 
		feed_dict_single = []
		gs, tvars = zip(*self._grads_and_vars)  

		# Compute the bias correction term.  
		bias_correction = sess.run(gs, 
			feed_dict=dict(zip(input_ph, input_data))) 

		# Initialize the feed_dict for each mini-batch update.
		for i in range(int(batch_size/mini_batchsize)):
			feed_dict_single.append({input_ph[j]:\
			input_data[j][i*mini_batchsize: (i + 1) * mini_batchsize] for j in range(len(input_data))})

		# Carry out the update.
		for i in range(int(batch_size/mini_batchsize)):
			# Compute the gradient for each mini-batch update. 
			single_grads_and_vars = sess.run(gs, 
				feed_dict = feed_dict_single[i])  

			# Construct the feed dict for the mini batch update. 
			_feed_dict = feed_dict_single[i] 
			_feed_dict.update(dict(zip(self._bias_correction_ph, bias_correction)))  
			_feed_dict.update(dict(zip(self._variance_reduction_ph, single_grads_and_vars)))

			_ = sess.run(self._update,
			feed_dict = _feed_dict)






