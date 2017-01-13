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

class dist_rnn(object):

    def __init__(self, forecast=50, pa=None, offset=1):

        #self.INPUT_DIM = 1
        self.INPUT_DIM = 1
        self.EMBEDDING_SIZE = 8
        self.CELL_DIM = 16
        self.N_SEQS = 100000
        self.N_PRINT = 100
        self.BATCH_SIZE = 64
        self.SEQ_LEN = 50
        self.NUM_TRAIN_STEP = 100000
        self.LEARNING_RATE = 0.01
        self.OFFSET = offset
        self.FORECAST_LEN = forecast

        self.trained = False
        self.training = False

        if pa is None:
            pa = parameters.Parameters()
            pa.dist.periodic = True
            pa.dist.bimodal = False
            pa.dist.noise = True

        pa.simu_len = self.SEQ_LEN + self.OFFSET

        self.pa = pa
        self.inputs = [generate_sequence_for_rnn(pa)[:,0] for _ in range(self.N_SEQS + 1)]

        self.build_graph()

    def build_graph(self):
        print('Build model...')

        # print(inputs)

        self.data_sequence = [tf.placeholder(tf.int32, shape=[self.BATCH_SIZE])
                         for _ in range(self.SEQ_LEN + self.OFFSET)]

        scaled_ds = [d - 1 for d in self.data_sequence]

        self.embeddings = tf.Variable(tf.random_normal([self.pa.max_job_len, self.EMBEDDING_SIZE], stddev=0.35),
                                        name="embeddings")

        self.embedded_data_sequence = [tf.nn.embedding_lookup(self.embeddings, d) for d in scaled_ds]

        print(self.embedded_data_sequence)

        # expanded_data_sequence = [tf.expand_dims(d, dim=1) for d in self.data_sequence]

        x_seq = self.embedded_data_sequence[:-self.OFFSET]

        y_seq = scaled_ds[self.OFFSET:]

        self.single_data_sequence = [tf.placeholder(tf.int32, shape=[1])
                         for _ in range(self.SEQ_LEN)]

        scaled_single_ds = [d - 1 for d in self.single_data_sequence]

        self.embedded_single_data_sequence = [tf.nn.embedding_lookup(self.embeddings, d) for d in scaled_single_ds]

        lstm = tf.nn.rnn_cell.LSTMCell(self.CELL_DIM, state_is_tuple=False)
        # Initial state of the LSTM memory.
        # print lstm.state_size
        state = lstm.zero_state(self.BATCH_SIZE, tf.float32)
        pred_w = tf.Variable(tf.random_normal([self.CELL_DIM, self.pa.max_job_len], stddev=0.35),
                              name="weights")
        pred_b = tf.Variable(tf.zeros([self.pa.max_job_len]), name="biases")
        predictions = []

        global_step = tf.Variable(0)
        loss = 0.0
        with tf.variable_scope("state_saving_lstm") as scope:
            for current_input_batch, current_target_batch in zip(x_seq, y_seq):
                # The value of state is updated after processing each batch of words.

                output, state = lstm(current_input_batch, state)

                # The LSTM output can be used to make next word predictions
                logits = tf.matmul(output, pred_w) + pred_b
                #print(pred)
                #print(current_target_batch)
                pred = tf.multinomial(logits, 1)
                pred = tf.squeeze(pred)
                predictions.append(pred)
                loss += tf.reduce_mean(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, current_target_batch)))
                scope.reuse_variables()

            self.loss = loss

            opt = tf.train.RMSPropOptimizer(tf.train.exponential_decay(self.LEARNING_RATE, global_step, 1000, 0.95, staircase=True))
            self.train_step = opt.minimize(loss, global_step=global_step)

            state = lstm.zero_state(1, tf.float32)
            generated_guided_seq = []
            for i in range(int(np.floor(self.SEQ_LEN/2.0))):
                x_init = tf.expand_dims(x_seq[i][0], 0)
                output, state = lstm(x_init, state)
                logits = tf.matmul(output, pred_w) + pred_b
                pred = tf.multinomial(logits, 1)
                pred = tf.squeeze(pred)
                generated_guided_seq.append(pred)


            for _ in range(int(np.ceil(self.SEQ_LEN/2.0))):
                x = tf.expand_dims(tf.nn.embedding_lookup(self.embeddings, pred), 0)
                output, state = lstm(x, state)
                logits = tf.matmul(output, pred_w) + pred_b
                pred = tf.multinomial(logits, 1)
                pred = tf.squeeze(pred)
                generated_guided_seq.append(pred)

            self.generated_guided_seq = generated_guided_seq

            generated_seq = []
            state = lstm.zero_state(1, tf.float32)
            for x in self.embedded_single_data_sequence:
                output, state = lstm(x, state)
                logits = tf.matmul(output, pred_w) + pred_b
                pred = tf.multinomial(logits, 1)
                pred = tf.squeeze(pred)
                generated_seq.append(pred)

            self.generated_seq = generated_seq

            state = lstm.zero_state(1, tf.float32)
            for x in self.embedded_single_data_sequence:
                output, state = lstm(x, state)
                logits = tf.matmul(output, pred_w) + pred_b
                pred = tf.multinomial(logits, 1)
                pred = tf.squeeze(pred)

            generated_forecast_seq = [pred]

            for _ in range(self.FORECAST_LEN):
                x = tf.expand_dims(tf.nn.embedding_lookup(self.embeddings, pred), 0)
                output, state = lstm(x, state)
                logits = tf.matmul(output, pred_w) + pred_b
                pred = tf.multinomial(logits, 1)
                pred = tf.squeeze(pred)
                generated_forecast_seq.append(pred)

            self.forecast = generated_forecast_seq


        print("Initializing...")
        init = tf.initialize_all_variables()

        self.sess = tf.Session()
        self.sess.run(init)

    def train(self):
        self.training = True
        self.trained = False
        print("Training RNN...")
        # train the model, output generated text after each iteration
        for iteration in range(self.NUM_TRAIN_STEP):

            print()
            print('-' * 50)
            print('Iteration', iteration)
            #print(inputs)
            seqs = [self.inputs[random.randint(0, self.N_SEQS - 1)] for _ in range(self.BATCH_SIZE)]
            # print(seqs)
            self.data = [np.asarray([seqs[i][j] for i in range(self.BATCH_SIZE)]) for j in range(self.SEQ_LEN + self.OFFSET)]
            print('----- training:')
            # print(data)
            #print(data_sequence)
            _, l, generated = self.sess.run([self.train_step, self.loss, self.generated_guided_seq], feed_dict=dict(zip(self.data_sequence, self.data)))
            print("Error: " + str(l))
            if iteration % 500 == 0:
                print('----- ground truth:')
                print([d[:][0] for d in self.data])
                print('----- generating:')
                print([g + 1 for g in generated])
            #print(sess.run(pred_b))

            if iteration % 1250 == 0:
                print('----- starter seq:')
                print([d[:][0] for d in self.data])
                print('----- forecast')
                print(self.sample_forecast_with_starter_seq([d[:][0] for d in self.data]))

        self.trained = True
        self.training = False

    def get_predictions_for_seq(self, seq):
        if not self.trained:
            raise Exception("RNN has not been trained")

        predictions = self.sess.run(self.generated_seq, feed_dict=dict(zip(self.single_data_sequence, seq)))

        return predictions

    def sample_forecast_with_starter_seq(self, seq):
        if not (self.trained or self.training):
            raise Exception("RNN has not been trained")

        forecast = self.sess.run(self.forecast, feed_dict=dict(zip(self.single_data_sequence, [[s] for s in seq])))
        return forecast




if __name__ == '__main__':
    model = dist_rnn(offset=1)
    model.train()