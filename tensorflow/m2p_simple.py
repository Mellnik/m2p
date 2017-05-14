'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow..
Next word prediction after n_input words learned from text file.
A story is automatically generated if the predicted word is fed back as input.

Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
import json

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


# Target log path
logs_path = '.\\logs'
writer = tf.summary.FileWriter(logs_path)

# json file containg trading data
training_file = '..\\BTC_ETH_2weeks.json'

with open(training_file) as data_file:
	training_data = json.load(data_file)

training_data_total = len(training_data)

print("Loaded", training_data_total, "trading data points from", training_file)

vocab_size = 2 # total number of possible unique inputs (112)

# Parameters
learning_rate = 0.001
training_iters = 50000 # wie oft trainiert werden soll....
display_step = 50 # output status all 1000 iterations
n_input = 24 # LSTM takes 3 inputs = 3 words
'''
[[[ 52]
  [ 34]
  [107]]]
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
   0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
   0.  0.  0.  0.]]
  '''

# number of units in RNN cell
n_hidden = 512 # hidden layer num of features ???

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1]) # die daten zum trainieren ???
y = tf.placeholder("float", [None, vocab_size]) # Anzahl der einzigartigen inputs / outputs ???

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size])) # 512, 112
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size])) # Anzahl der einzigarten inputs / outputs ???
}

def RNN(x, weights, biases):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)
    print(x)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)

	# ca 210 woerter sind in der datei, 3 werden pro loop trainiert, d.h. nach 70 loops hat er die datei
	# offset = offset + n_input + 1    <-- pro loop
	# wenn offset groesser als die 210 woerter datei -> reset auf 0 bis n_input + 1
	# dieser eingefuehrte Zufall hilft, dass die trainierten sequenzen nicht immer die gleichen sind (nice)
	# d.h. bei 4032 trading data und 50k iterations, wuerde er meine 2weeks.json ca 12.4x durchlaufen
    while step < training_iters:
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(training_data) - end_offset):
            offset = random.randint(0, n_input + 1)

	    # 3 words
        symbols_in_keys = [ [training_data[i]['close'] - training_data[i]['open']] for i in range(offset, offset + n_input) ]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
		
		# the word after the 3 words which is used as output label		
        index_after = offset + n_input
        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        if training_data[index_after]['close'] - training_data[index_after]['open'] > 0:
            symbols_out_onehot[0] = 1.0
        else:
            symbols_out_onehot[1] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        loss_total += loss
        acc_total += acc
        if (step+1) % display_step == 0:
            print("Iter= " + str(step + 1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
        step += 1
        offset += (n_input+1)
    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))
    print("Run on command line.")
    print("\ttensorboard --logdir=%s" % (logs_path))
    print("Point your web browser to: http://localhost:6006/")

