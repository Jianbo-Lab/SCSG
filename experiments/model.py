"""
This script contains functions that build graphs for the experiment. 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

NUM_CLASSES = 10 
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE 

def build_network(images, model = 'cnn'):
	"""
	Build network for the model.
	"""
	if model == 'cnn':
		net = tf.reshape(images, [-1,IMAGE_SIZE,IMAGE_SIZE,1])
		net = slim.conv2d(net, num_outputs=32, kernel_size=5, scope='conv1')
		net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
		net = slim.conv2d(net, num_outputs=64, kernel_size=5, scope='conv2')
		net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
		net = slim.flatten(net)
		net = slim.fully_connected(net, 1024, scope='fully_connected4')
		net = slim.fully_connected(net, 10, activation_fn=None, 
			scope='fully_connected5')
	elif model == 'fc':
		net = tf.reshape(images,[-1,IMAGE_SIZE,IMAGE_SIZE,1])
		net = slim.flatten(net)
		net = slim.fully_connected(net, 512, scope='fully_connected1')
		net = slim.fully_connected(net, 512, scope='fully_connected2')
		net = slim.fully_connected(net, 512, scope='fully_connected3')
		net = slim.fully_connected(net, 10, activation_fn=None, 
			scope='fully_connected4')
	return net

def loss(logits, labels):
	"""
	Calculates the loss from the logits and the labels.
	Args:
	logits: [batch_size, NUM_CLASSES].
	labels: [batch_size].
	Returns:
	loss: Loss tensor 
	""" 
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
		labels=labels, logits=logits, name='xentropy')
	return cross_entropy


def training(loss, learning_rate):
	"""
	Sets up the training Ops. 
	Args:
	loss: Loss tensor 
	learning_rate: The learning rate for gradient descent.
	Returns:
	train_op: The Op for training.
	"""  
	optimizer = tf.train.GradientDescentOptimizer(learning_rate) 
	global_step = tf.Variable(0, name='global_step', trainable=False) 
	train_op = optimizer.minimize(loss, global_step=global_step)
	return train_op


def evaluation(logits, labels):
	"""Evaluate the quality of the logits at predicting the label.
	Args:
	logits: [batch_size, NUM_CLASSES].
	labels: [batch_size] 
	Returns: accuracy
	""" 
	correct = tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))
	# Return accuracy
	return tf.reduce_mean(tf.cast(correct, tf.float32))
