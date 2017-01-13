import numpy
import matplotlib.pyplot as plt
import pandas
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim

import random
import sys
from job_distribution import *
import parameters

INPUT_DIM = 1
CELL_DIM = 16
N_SEQS = 100000
N_PRINT = 100
BATCH_SIZE = 16
SEQ_LEN = 50
NUM_TRAIN_STEP = 10000
LEARNING_RATE = 0.01

pa = parameters.Parameters()
pa.simu_len = SEQ_LEN + 1
pa.dist.periodic = True
pa.dist.bimodal = False
pa.dist.noise = True
inputs = [generate_sequence_for_rnn(pa)[:,0] for _ in range(N_SEQS + 1)]

print('Build model...')

# print(inputs)

data_sequence = [tf.placeholder(tf.float32, shape=[BATCH_SIZE])
                 for _ in range(SEQ_LEN + 1)]

expanded_data_sequence = [tf.expand_dims(d, dim=1) for d in data_sequence]

x_seq = expanded_data_sequence[:-1]
y_seq = expanded_data_sequence[1:]

lstm = tf.nn.rnn_cell.LSTMCell(CELL_DIM, state_is_tuple=False)
# Initial state of the LSTM memory.
# print lstm.state_size
state = lstm.zero_state(BATCH_SIZE, tf.float32)
pred_w = tf.Variable(tf.random_normal([CELL_DIM, INPUT_DIM], stddev=0.35),
                      name="weights")
pred_b = tf.Variable(tf.zeros([INPUT_DIM]), name="biases")
predictions = []

global_step = tf.Variable(0)
loss = 0.0
with tf.variable_scope("state_saving_lstm") as scope:
    for current_input_batch, current_target_batch in zip(x_seq, y_seq):
        # The value of state is updated after processing each batch of words.
        output, state = lstm(current_input_batch, state)

        # The LSTM output can be used to make next word predictions
        pred = tf.matmul(output, pred_w) + pred_b
        #print(pred)
        #print(current_target_batch)
        predictions.append(pred)
        loss += tf.reduce_mean(tf.reduce_mean(tf.nn.l2_loss(pred - current_target_batch)))
        scope.reuse_variables()

    opt = tf.train.RMSPropOptimizer(tf.train.exponential_decay(LEARNING_RATE, global_step, 1000, 0.95, staircase=True))
    train_step = opt.minimize(loss, global_step=global_step)

    state = lstm.zero_state(1, tf.float32)
    for i in range(int(np.floor(SEQ_LEN/2.0))):
        x_init = tf.expand_dims(x_seq[i][0, :], dim=0)
        output, state = lstm(x_init, state)
        pred = tf.matmul(output, pred_w) + pred_b

    generated_seq = [pred]

    for _ in range(int(np.ceil(SEQ_LEN/2.0))):
        output, state = lstm(pred, state)
        pred = tf.matmul(output, pred_w) + pred_b
        generated_seq.append(pred)

print("Initializing...")
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print("Training...")
# train the model, output generated text after each iteration
for iteration in range(NUM_TRAIN_STEP):

    print()
    print('-' * 50)
    print('Iteration', iteration)
    #print(inputs)
    seqs = [inputs[random.randint(0, N_SEQS - 1)] for _ in range(BATCH_SIZE)]
    # print(seqs)
    data = [np.asarray([seqs[i][j] for i in range(BATCH_SIZE)]) for j in range(SEQ_LEN + 1)]
    print('----- training:')
    # print(data)
    #print(data_sequence)
    _, l, generated = sess.run([train_step, loss, generated_seq], feed_dict=dict(zip(data_sequence, data)))
    print("Error: " + str(l))
    if iteration % 500 == 0:
        print('----- ground truth:')
        print([d[:][0] for d in data[int(np.floor(SEQ_LEN/2)):]])
        print('----- generating:')
        print([g[0][0] for g in generated])
    #print(sess.run(pred_b))