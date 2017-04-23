#Simple softmax regression model for MNIST data

import numpy as numpy
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
import input_data

data_dir = '/cis/home/ajacob/Documents/enlitic/data/'

mnist = input_data.read_data_sets(data_dir, one_hot=True)
sample_img = mnist.train.next_batch(1)

nFeatures =  sample_img[0].shape[1]
nLabels = sample_img[1].shape[1]

x_ = tf.placeholder(tf.float32,shape = [None,nFeatures])
y_true = tf.placeholder(tf.float32,shape = [None,nLabels])

weights = tf.Variable(tf.truncated_normal([nFeatures,nLabels],stddev = 0.1))
biases = tf.Variable(tf.zeros([1,nLabels]))

logits = tf.matmul(x_,weights) + biases
pred = tf.nn.softmax(logits)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_true))

correct_pred = tf.equal(tf.arg_max(pred,1),tf.arg_max(y_true,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

trainer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cross_entropy)



nEpochs = 1000
batchSize = 500
i = 1

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer()) #Initialize all variables

	while(i < nEpochs):

		train_imgs,train_labels = mnist.train.next_batch(batchSize)
		test_imgs,test_labels = mnist.test.next_batch(100)


		trainer.run(feed_dict = {x_ : train_imgs,y_true : train_labels})

		acc_train = accuracy.eval(feed_dict = {x_ : train_imgs,y_true : train_labels})
		print('Epoch ' + str(i) + ': Train accuracy = ' + str(acc_train))

		acc_test = accuracy.eval(feed_dict = {x_ : test_imgs,y_true : test_labels})
		print('Verification accuracy = ' + str(acc_test))

		i += 1




