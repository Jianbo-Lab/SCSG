"""
This script runs the experiments for comparing SCSG and SGD on MNIST.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math,argparse,time

import numpy as np 
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import sys
sys.path.append('../core') 
from scsg import *
from model import * 

def run_single_experiment(method='sgd',
				 model = 'cnn',
				 batch_size = 100, 
				 learning_rate = 0.001,
				 ratio = 10,
				 num_iterations = 400, 
				 fix_batch = False):
	"""
	Carries out a single experiment of training MNIST. It saves the  
	Args:
	method: str. optimization method to use: sgd or scsg.
	model: str. use cnn or fully connected network.
	batch_size: int. batch size for training. (Only used for fixed batchsize experiments.)
	learning rate: float. learning rate for training. 
	ratio: the ratio of batch size and mini-batch size. (Only for SCSG)
	num_iterations: int. Number of iterations for training.
	fix_batch: bool. If the batchsize is fixed or increasing according to the scheme in paper.  
	Return:
	None
	"""

	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	# Build placeholders and networks. 
	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	logits = build_network(x, model = model)
	batch_loss = loss(logits, y_)
	loss_op = tf.reduce_mean(batch_loss) 
	acc_op = evaluation(logits, y_)

	if method == 'sgd':
		train_op = training(loss_op, learning_rate) 
	elif method == 'scsg':
		optimizer = SCSGOptimizer(loss_op, learning_rate) 

	print('Network constructed.')

	num_used_data = 0
	start_time = time.time()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for j in range(num_iterations):
			if not fix_batch: 
				batch_size = int((j+1) ** 1.5)

			mini_batchsize = max(1,int(batch_size / float(ratio))) # mini-batch size. 
			batch = mnist.train.next_batch(batch_size)
			feed_dict = {x:batch[0],y_:batch[1]}
			if method == 'scsg':
				optimizer.batch_update(sess,feed_dict,
								   batch_size,mini_batchsize, 
								   lr = 1.0/(j+1) if not fix_batch else None) 
			elif method == 'sgd':   
				_ = sess.run(train_op,
						 feed_dict=feed_dict) 
			num_used_data += batch_size
			if (j % 3 == 0 and (not fix_batch)) or (j % (num_iterations// 10) == 0 and fix_batch):
				# Record loss and accuracy every 3 iterations.
				samples = np.random.choice(range(mnist.train._num_examples),
					size = 10000, replace = False)
				train_loss_val = sess.run(loss_op,
							feed_dict = {x: mnist.train.images[samples], 
								 y_: mnist.train.labels[samples]})
				val_loss_val, acc_val = sess.run([loss_op,acc_op],
							feed_dict = {x: mnist.validation.images, 
								 y_: mnist.validation.labels}) 

				train_loss_val, val_loss_val, acc_val = \
				round(train_loss_val,3), round(val_loss_val,3), round(acc_val,3)

				elapsed_time = round(time.time() - start_time, 2) 
				print('# iteration: {} # data used: {} Time elapsed: {} \n Train loss: {} Val loss: {} Val Accuracy: {}'.format(j, num_used_data, elapsed_time, train_loss_val, val_loss_val, acc_val))
				print('----------------------------------------') 


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model',type = str,default = 'fc', choices = ['cnn', 'fc']) 
	parser.add_argument('--method',type = str,default = 'scsg',
		choices = ['scsg', 'sgd']) 
	parser.add_argument('--batch_size',type=int, default = 1000)
	parser.add_argument('--num_iterations',type=int, default = 400) 
	parser.add_argument('--learning_rate',type=float, default = 0.1)
	parser.add_argument('--ratio',type=int,default=32) 
	parser.add_argument('--fix_batch', action='store_true')
	args = parser.parse_args() 

	run_single_experiment(method = args.method, 
		model = args.model, 
		batch_size = args.batch_size,
		learning_rate = args.learning_rate, 
		ratio = args.ratio,
		num_iterations = args.num_iterations,
		fix_batch = args.fix_batch,
		) 

if __name__ == '__main__': 
	main()
