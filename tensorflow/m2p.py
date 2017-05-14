from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
import json

print("Market To Probability (M2P)")

start_time = time.time()
def elapsed(sec):
    if sec < 60:
        return str(sec) + " sec"
    elif sec < (60 * 60):
        return str(sec / 60) + " min"
    else:
        return str(sec / (60 * 60)) + " hr"

# For TensorBoard logs
logs_path = '.\\logs'
writer = tf.summary.FileWriter(logs_path)

# json file containg trading data
training_file = '..\\BTC_ETH_2weeks.json'

with open(training_file) as data_file:
	training_data = json.load(data_file)

training_data_total = len(training_data)

print("Loaded", training_data_total, "trading data points from", training_file)

#for i in data:
	#print(i['date'], i['open'])
	
# 4032 data points = 2 weeks
# 72 data points = 6 hours = 1 data period
learning_rate = 0.001 #fuer den RMSProp
training_iters = 20000 # wie oft trainiert werden soll....
display_step = 50 # output status all 1000 iterations
n_input = 72 # open, close, volume
#n_steps = 72 # multiple data points in one row, each data points represents 300 seconds, 72 * 300 = 6 hrs
n_hidden = 256 # number of units in RNN cell # hidden layer num of features ???
n_classes = 2 # 0-1 = Buy or Sell

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1]) # [Batch Size, Sequence Length, Input Dimension]
y = tf.placeholder("float", [None, n_classes])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes])) # 1024, 2
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes])) # 2
}

def RNN(x, weights, biases):
	#x = tf.unstack(x, n_steps, 1)
	x = tf.reshape(x, [-1, n_input])
	x = tf.split(x, n_input, 1)
	
	stacked_lstm = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])
	#lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
	
	outputs, states = rnn.static_rnn(stacked_lstm, x, dtype=tf.float32)
	
	return tf.matmul(outputs[-1], weights['out']) + biases['out']
	
pred = RNN(x, weights, biases)

print("Setting up engine...")

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	step = 0
	offset = random.randint(0, n_input + 1)
	end_offset = n_input + 1
	loss_total = 0
	acc_total = 0
	
	writer.add_graph(sess.graph)
	
	print("Entering training loop...")
	
	while step < training_iters:
		if offset >= (training_data_total - end_offset):
			offset = random.randint(0, n_input + 1)
		
		trading_data = []
		for i in range(offset, offset + n_input):
			trading_data.append([training_data[i]['quoteVolume']]) #'''training_data[i]['open'], training_data[i]['close'], '''
		#trading_data = [trading_data]
		trading_data = np.reshape(np.array(trading_data), [-1, n_input, 1])
			
		index_after = offset + n_input
		trading_data_onehot = np.zeros([n_classes], dtype=float)
		if training_data[index_after]['close'] - training_data[index_after]['open'] > 0:
			trading_data_onehot[0] = 1.0
		else:
			trading_data_onehot[1] = 1.0
		trading_data_onehot = np.reshape(trading_data_onehot, [1, -1])
		
		#print(trading_data, len(trading_data))
		#print(trading_data_onehot, len(trading_data_onehot))
		#print("ONEHOT: ", trading_data_onehot, training_data[index_after]['close'], training_data[index_after]['open'], training_data[index_after]['close'] - training_data[index_after]['open'], training_data[index_after]['quoteVolume'])
		#sys.exit(1)
	
		_, acc, loss, onehot_pred = sess.run([optimizer, accuracy, cost, pred], feed_dict={x: trading_data, y: trading_data_onehot})
		
		loss_total += loss
		acc_total += acc
		if (step + 1) % display_step == 0:
			print("Iter= " + str(step + 1) + ", Average Loss= " + "{:.6f}".format(loss_total / display_step) + ", Average Accuracy= " + "{:.2f}%".format(100 * acc_total / display_step))
			acc_total = 0
			loss_total = 0
		step += 1
		offset += (n_input + 1)

	print("Training complete!")
	print("Elapsed time: ", elapsed(time.time() - start_time))























