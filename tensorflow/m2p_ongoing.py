from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import urllib.request
import time
import json
import math
import sys

print("Market To Probability (M2P)")

if len(sys.argv) != 3:
    print("Usage:", sys.argv[0], "<date_start> <num_data_points>")
    sys.exit(1)

date_start = int(sys.argv[1])
num_data_points = int(sys.argv[2])
date_end = date_start + ((num_data_points - 1) * 300)
	
print("Starting date: ", date_start, " End date: ", date_end)

def elapsed(sec):
    if sec < 60:
        return str(sec) + " sec"
    elif sec < (60 * 60):
        return str(sec / 60) + " min"
    else:
        return str(sec / (60 * 60)) + " hr"

start_time = time.time()

# Target log path
logs_path = '.\\logs'
writer = tf.summary.FileWriter(logs_path)

# json file containg trading data
training_file = "https://poloniex.com/public?command=returnChartData&currencyPair=BTC_ETH&start=" + str(date_start) + "&end=" + str(date_end) + "&period=300"
with urllib.request.urlopen(training_file) as url:
    training_data = json.loads(url.read().decode())
    training_data_total = len(training_data)

print("Loaded", training_data_total, "trading data points from", training_file)
print("First entry: ", training_data[0]['date'], ", last entry", training_data[training_data_total - 1]['date'])

def build_dataset(words):
    dictionary = dict([('buy', 0), ('sell', 1), ('hold', 2)]);
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary
	
dictionary, reverse_dictionary = build_dataset(training_data) # all options in reverse, why??
vocab_size = len(dictionary) # total number of possible unique inputs (3)

def WhatToDo(deltaP):
    if deltaP > 0.0:
        return 'buy'
    else:
        return 'sell'

# Parameters
learning_rate = 0.001 # how fast to learn
training_iters = 20000 # how often to train
display_step = 1000 # output status all 100 iterations
n_input = 12 # LSTM takes 12 inputs = 12x buy/sell/hold

# number of units in RNN cell
n_hidden = 128 # hidden layer num of features ???

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1]) # die inputdaten zum trainieren, n_input * 1, ist ein array, 1 weil das arrray eindimensional sind
y = tf.placeholder("float", [None, vocab_size]) # Anzahl der einzigartigen outputs

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size])) # Anzahl der einzigarten inputs / outputs ???
}

def RNN(x, weights, biases):
    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x, n_input, 1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Loss and optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost_function)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

print("Starting session with", training_iters, "iterations...")

with tf.Session() as session:
    session.run(init_op)
    exit_training = False
    step = 0
    offset = random.randint(0, n_input + 1)
    end_offset = n_input + 1
    acc_final = 0
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)

    #saver.restore(session, ".\\models\\model_96dp_v1.ckpt")
    #print("Model restored.")
	
    while step < training_iters and exit_training == False:
        #if step == 4:
            #sys.exit(1)
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(training_data) - end_offset):
            offset = random.randint(0, n_input + 1)

	    # 3 words
        symbols_in_keys = [ [dictionary[ WhatToDo(training_data[i]['close'] - training_data[i]['open'])]] for i in range(offset, offset + n_input) ]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
        #print(symbols_in_keys)
        #for i in range(offset, offset + n_input):
            #print(training_data[i]['date'])
        
        # the word after the 3 words which is used as output label
        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[ WhatToDo(training_data[offset + n_input]['close'] - training_data[offset + n_input]['open'])]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])
        #print(symbols_out_onehot)

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost_function, pred], feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
		
        loss_total += loss
        acc_total += acc
        if (step + 1) % display_step == 0:
            acc_final = (100 * acc_total / display_step)
            print("Iter= " + str(step + 1) + ", Average Loss= " + "{:.6f}".format(loss_total / display_step) + ", Average Accuracy= " + "{:.2f}%".format(100 * acc_total / display_step))
            if (100 * acc_total / display_step) >= 100:
                exit_training = False
            acc_total = 0
            loss_total = 0
        step += 1
        offset += (n_input + 1)
    print("Optimization finished! Reached " + "{:.2f}%".format(acc_final))
    print("Elapsed time: ", elapsed(time.time() - start_time))
    save_path = saver.save(session, ".\\models_ongoing\\model_" + str(date_start) + "_" + str(date_end) + ".ckpt")
    print("Model saved in file: %s" % save_path)
    datapoint_index = training_data_total - n_input - 1
    result_set = ""
    date_count = 0
    future_file = "https://poloniex.com/public?command=returnChartData&currencyPair=BTC_ETH&start=" + str(training_data[int(datapoint_index) + n_input]['date']) + "&end=" + str(training_data[int(datapoint_index) + n_input]['date'] + 5 * 300) + "&period=300"
    print(future_file)
    with urllib.request.urlopen(future_file) as url:
        future_data = json.loads(url.read().decode())
    #print(future_data)
    symbols_in_keys = [dictionary[WhatToDo(training_data[i]['close'] - training_data[i]['open'])] for i in range(int(datapoint_index), int(datapoint_index) + n_input) ]
    for i in range(5):
        keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
        onehot_pred = session.run(pred, feed_dict={x: keys})
        onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
        if (future_data[i]['close'] - future_data[i]['open']) > 0:
            real_pred = 0
        else:
            real_pred = 1
        result_set = "%s, {%s onehot_pred %s %i, real_pred %i}" % (result_set, training_data[int(datapoint_index) + n_input]['date'] + date_count, reverse_dictionary[onehot_pred_index], onehot_pred_index, real_pred)
        date_count += 300
        symbols_in_keys = symbols_in_keys[1:]
        symbols_in_keys.append(onehot_pred_index)
    print(result_set)
