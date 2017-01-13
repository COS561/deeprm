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


'''
USAGE:

    model = dist_rnn(periodic=True, noise=False)
    model.train()
    model.sample_forecast_with_starter_seq(job_len_seq, job_size1_seq, job_size2_seq)

'''

class dist_rnn(object):

    def __init__(self, forecast=50, seq_len=50, pa=None, offset=1, noise=False, bimodal=False, periodic=True):

        #self.INPUT_DIM = 1
        self.INPUT_DIM = 1
        self.EMBEDDING_SIZE = 4
        self.CELL_DIM = 32
        self.N_SEQS = 100000
        self.N_PRINT = 100
        self.BATCH_SIZE = 32
        self.SEQ_LEN = seq_len
        self.NUM_TRAIN_STEP = 100000
        self.LEARNING_RATE = 0.01
        self.OFFSET = offset
        self.FORECAST_LEN = forecast

        self.trained = False
        self.training = False


        if pa is None:
            pa = parameters.Parameters()
            pa.dist.periodic = periodic
            pa.dist.bimodal = bimodal
            pa.dist.noise = noise

        print pa.max_job_len
        print pa.max_job_size
        print pa.dist.max_nw_size

        pa.simu_len = self.SEQ_LEN + self.OFFSET

        self.pa = pa
        self.len_inputs, self.s1_inputs, self.s2_inputs = self.get_inputs(pa, self.N_SEQS)

        self.build_graph()

    def get_inputs(self, pa, num_ex):
        tmp_num_ex = pa.num_ex
        pa.num_ex = num_ex
        len_seq, size_seq = generate_sequence_work(pa, seed=None)
        size1_seq = size_seq[:,:,0]
        size2_seq = size_seq[:,:,1]
        pa.num_ex = tmp_num_ex
        return len_seq, size1_seq, size2_seq


    def build_graph(self):
        print('Build model...')

        # print(inputs)

        self.len_sequence = [tf.placeholder(tf.int32, shape=[self.BATCH_SIZE])
                         for _ in range(self.SEQ_LEN + self.OFFSET)]
        scaled_len = [d - 1 for d in self.len_sequence]

        self.len_embeddings = tf.Variable(tf.random_normal([self.pa.max_job_len, self.EMBEDDING_SIZE], stddev=0.35),
                                        name="embeddings")

        self.embedded_len_sequence = [tf.nn.embedding_lookup(self.len_embeddings, d) for d in scaled_len]

        print(self.embedded_len_sequence)

        self.s1_sequence = [tf.placeholder(tf.int32, shape=[self.BATCH_SIZE])
                         for _ in range(self.SEQ_LEN + self.OFFSET)]
        scaled_s1 = [d - 1 for d in self.s1_sequence]

        self.s1_embeddings = tf.Variable(tf.random_normal([self.pa.dist.max_nw_size, self.EMBEDDING_SIZE], stddev=0.35),
                                        name="embeddings")

        self.embedded_s1_sequence = [tf.nn.embedding_lookup(self.s1_embeddings, d) for d in scaled_s1]

        print(self.embedded_s1_sequence)

        self.s2_sequence = [tf.placeholder(tf.int32, shape=[self.BATCH_SIZE])
                         for _ in range(self.SEQ_LEN + self.OFFSET)]
        scaled_s2 = [d - 1 for d in self.s2_sequence]

        self.s2_embeddings = tf.Variable(tf.random_normal([self.pa.dist.max_nw_size, self.EMBEDDING_SIZE], stddev=0.35),
                                        name="embeddings")

        self.embedded_s2_sequence = [tf.nn.embedding_lookup(self.s2_embeddings, d) for d in scaled_s2]

        print(self.embedded_s2_sequence)

        # expanded_data_sequence = [tf.expand_dims(d, dim=1) for d in self.data_sequence]

        len_x_seq = self.embedded_len_sequence[:-self.OFFSET]
        len_y_seq = scaled_len[self.OFFSET:]

        s1_x_seq = self.embedded_s1_sequence[:-self.OFFSET]
        s1_y_seq = scaled_s1[self.OFFSET:]

        s2_x_seq = self.embedded_s2_sequence[:-self.OFFSET]
        s2_y_seq = scaled_s2[self.OFFSET:]


        self.single_len_sequence = [tf.placeholder(tf.int32, shape=[1])
                         for _ in range(self.SEQ_LEN)]
        self.single_s1_sequence = [tf.placeholder(tf.int32, shape=[1])
                         for _ in range(self.SEQ_LEN)]
        self.single_s2_sequence = [tf.placeholder(tf.int32, shape=[1])
                         for _ in range(self.SEQ_LEN)]

        scaled_single_len = [d - 1 for d in self.single_len_sequence]
        scaled_single_s1 = [d - 1 for d in self.single_s1_sequence]
        scaled_single_s2 = [d - 1 for d in self.single_s2_sequence]


        self.embedded_single_len_sequence = [tf.nn.embedding_lookup(self.len_embeddings, d) for d in scaled_single_len]
        self.embedded_single_s1_sequence = [tf.nn.embedding_lookup(self.s1_embeddings, d) for d in scaled_single_s1]
        self.embedded_single_s2_sequence = [tf.nn.embedding_lookup(self.s2_embeddings, d) for d in scaled_single_s2]

        lstm = tf.nn.rnn_cell.LSTMCell(self.CELL_DIM, state_is_tuple=False)
        # Initial state of the LSTM memory.
        # print lstm.state_size
        state = lstm.zero_state(self.BATCH_SIZE, tf.float32)
        len_pred_w = tf.Variable(tf.random_normal([self.CELL_DIM, self.pa.max_job_len], stddev=0.35),
                              name="weights")
        len_pred_b = tf.Variable(tf.zeros([self.pa.max_job_len]), name="biases")
        s1_pred_w = tf.Variable(tf.random_normal([self.CELL_DIM, self.pa.dist.max_nw_size], stddev=0.35),
                                 name="weights")
        s1_pred_b = tf.Variable(tf.zeros([self.pa.dist.max_nw_size]), name="biases")
        s2_pred_w = tf.Variable(tf.random_normal([self.CELL_DIM, self.pa.dist.max_nw_size], stddev=0.35),
                                 name="weights")
        s2_pred_b = tf.Variable(tf.zeros([self.pa.dist.max_nw_size]), name="biases")

        global_step = tf.Variable(0)
        loss = 0.0
        with tf.variable_scope("state_saving_lstm") as scope:
            for len_x, s1_x, s2_x, len_y, s1_y, s2_y in zip(len_x_seq, s1_x_seq, s2_x_seq, len_y_seq, s1_y_seq, s2_y_seq):
                # The value of state is updated after processing each batch of words.

                features = tf.concat(1, [len_x, s1_x, s2_x])

                output, state = lstm(features, state)

                # The LSTM output can be used to make next word predictions
                len_logits = tf.matmul(output, len_pred_w) + len_pred_b
                s1_logits = tf.matmul(output, s1_pred_w) + s1_pred_b
                s2_logits = tf.matmul(output, s2_pred_w) + s2_pred_b

                loss += tf.reduce_mean(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(len_logits, len_y)))
                loss += tf.reduce_mean(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(s1_logits, s1_y)))
                loss += tf.reduce_mean(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(s2_logits, s2_y)))

                scope.reuse_variables()

            self.loss = loss

            opt = tf.train.RMSPropOptimizer(tf.train.exponential_decay(self.LEARNING_RATE, global_step, 300, 0.95, staircase=True))
            self.train_step = opt.minimize(loss, global_step=global_step)

            state = lstm.zero_state(1, tf.float32)
            generated_guided_seq = []
            for i in range(int(np.floor(self.SEQ_LEN/2.0))):
                len_x_init = tf.expand_dims(len_x_seq[i][0], 0)
                s1_x_init = tf.expand_dims(s1_x_seq[i][0], 0)
                s2_x_init = tf.expand_dims(s2_x_seq[i][0], 0)

                features = tf.concat(1, [len_x_init, s1_x_init, s2_x_init])

                output, state = lstm(features, state)
                len_logits = tf.matmul(output, len_pred_w) + len_pred_b
                s1_logits = tf.matmul(output, s1_pred_w) + s1_pred_b
                s2_logits = tf.matmul(output, s1_pred_w) + s1_pred_b

                len_pred = tf.multinomial(len_logits, 1)
                len_pred = tf.squeeze(len_pred)
                s1_pred = tf.multinomial(s1_logits, 1)
                s1_pred = tf.squeeze(s1_pred)
                s2_pred = tf.multinomial(s2_logits, 1)
                s2_pred = tf.squeeze(s2_pred)
                generated_guided_seq.append([len_pred, s1_pred, s2_pred])


            for _ in range(int(np.ceil(self.SEQ_LEN/2.0))):
                len_x = tf.expand_dims(tf.nn.embedding_lookup(self.len_embeddings, len_pred), 0)
                s1_x = tf.expand_dims(tf.nn.embedding_lookup(self.s1_embeddings, s1_pred), 0)
                s2_x = tf.expand_dims(tf.nn.embedding_lookup(self.s2_embeddings, s2_pred), 0)

                features = tf.concat(1, [len_x, s1_x, s2_x])

                output, state = lstm(features, state)
                len_logits = tf.matmul(output, len_pred_w) + len_pred_b
                s1_logits = tf.matmul(output, s1_pred_w) + s1_pred_b
                s2_logits = tf.matmul(output, s1_pred_w) + s1_pred_b
                len_pred = tf.multinomial(len_logits, 1)
                len_pred = tf.squeeze(len_pred)
                s1_pred = tf.multinomial(s1_logits, 1)
                s1_pred = tf.squeeze(s1_pred)
                s2_pred = tf.multinomial(s2_logits, 1)
                s2_pred = tf.squeeze(s2_pred)
                generated_guided_seq.append([len_pred, s1_pred, s2_pred])

            self.generated_guided_seq = generated_guided_seq

            generated_seq = []
            state = lstm.zero_state(1, tf.float32)
            for len_x, s1_x, s2_x in zip(self.embedded_single_len_sequence, self.embedded_single_s1_sequence, self.embedded_single_s2_sequence):

                features = tf.concat(1, [len_x, s1_x, s2_x])

                output, state = lstm(features, state)
                len_logits = tf.matmul(output, len_pred_w) + len_pred_b
                s1_logits = tf.matmul(output, s1_pred_w) + s1_pred_b
                s2_logits = tf.matmul(output, s1_pred_w) + s1_pred_b
                len_pred = tf.multinomial(len_logits, 1)
                len_pred = tf.squeeze(len_pred)
                s1_pred = tf.multinomial(s1_logits, 1)
                s1_pred = tf.squeeze(s1_pred)
                s2_pred = tf.multinomial(s2_logits, 1)
                s2_pred = tf.squeeze(s2_pred)
                generated_seq.append([len_pred, s1_pred, s2_pred])

            self.generated_seq = generated_seq

            state = lstm.zero_state(1, tf.float32)
            for len_x, s1_x, s2_x in zip(self.embedded_single_len_sequence, self.embedded_single_s1_sequence, self.embedded_single_s2_sequence):

                features = tf.concat(1, [len_x, s1_x, s2_x])

                output, state = lstm(features, state)
                len_logits = tf.matmul(output, len_pred_w) + len_pred_b
                s1_logits = tf.matmul(output, s1_pred_w) + s1_pred_b
                s2_logits = tf.matmul(output, s1_pred_w) + s1_pred_b
                len_pred = tf.multinomial(len_logits, 1)
                len_pred = tf.squeeze(len_pred)
                s1_pred = tf.multinomial(s1_logits, 1)
                s1_pred = tf.squeeze(s1_pred)
                s2_pred = tf.multinomial(s2_logits, 1)
                s2_pred = tf.squeeze(s2_pred)

            generated_forecast_seq = [[len_pred, s1_pred, s2_pred]]

            for _ in range(self.FORECAST_LEN):
                len_x = tf.expand_dims(tf.nn.embedding_lookup(self.len_embeddings, len_pred), 0)
                s1_x = tf.expand_dims(tf.nn.embedding_lookup(self.s1_embeddings, s1_pred), 0)
                s2_x = tf.expand_dims(tf.nn.embedding_lookup(self.s2_embeddings, s2_pred), 0)

                features = tf.concat(1, [len_x, s1_x, s2_x])

                output, state = lstm(features, state)
                len_logits = tf.matmul(output, len_pred_w) + len_pred_b
                s1_logits = tf.matmul(output, s1_pred_w) + s1_pred_b
                s2_logits = tf.matmul(output, s1_pred_w) + s1_pred_b
                len_pred = tf.multinomial(len_logits, 1)
                len_pred = tf.squeeze(len_pred)
                s1_pred = tf.multinomial(s1_logits, 1)
                s1_pred = tf.squeeze(s1_pred)
                s2_pred = tf.multinomial(s2_logits, 1)
                s2_pred = tf.squeeze(s2_pred)
                generated_forecast_seq.append([len_pred, s1_pred, s2_pred])

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
            len_seqs = [self.len_inputs[random.randint(0, self.N_SEQS - 1)] for _ in range(self.BATCH_SIZE)]
            self.len_data = [np.asarray([len_seqs[i][j] for i in range(self.BATCH_SIZE)]) for j in range(self.SEQ_LEN + self.OFFSET)]

            s1_seqs = [self.s1_inputs[random.randint(0, self.N_SEQS - 1)] for _ in range(self.BATCH_SIZE)]
            self.s1_data = [np.asarray([s1_seqs[i][j] for i in range(self.BATCH_SIZE)]) for j in range(self.SEQ_LEN + self.OFFSET)]

            s2_seqs = [self.s2_inputs[random.randint(0, self.N_SEQS - 1)] for _ in range(self.BATCH_SIZE)]
            self.s2_data = [np.asarray([s2_seqs[i][j] for i in range(self.BATCH_SIZE)]) for j in range(self.SEQ_LEN + self.OFFSET)]
            print('----- training:')
            # print(data)
            #print(data_sequence)
            _, l, generated = self.sess.run([self.train_step, self.loss, self.generated_guided_seq],
                                            feed_dict=dict(
                                                zip(self.len_sequence, self.len_data) +
                                                zip(self.s1_sequence, self.s1_data) +
                                                zip(self.s2_sequence, self.s2_data)))
            print("Error: " + str(l))
            if iteration % 500 == 0:
                print('----------- len -------------')
                print('----- ground truth:')
                print([d[:][0] for d in self.len_data])
                print('----- generating:')
                print([g[0] + 1 for g in generated])
                print('----------- s1 -------------')
                print('----- ground truth:')
                print([d[:][0] for d in self.s1_data])
                print('----- generating:')
                print([g[1] + 1 for g in generated])
                print('----------- s2 -------------')
                print('----- ground truth:')
                print([d[:][0] for d in self.s2_data])
                print('----- generating:')
                print([g[2] + 1 for g in generated])
            #print(sess.run(pred_b))

            if iteration % 1250 == 0:
                print('----- starter seq:')
                print([d[:][0] for d in self.len_data])
                print([d[:][0] for d in self.s1_data])
                print([d[:][0] for d in self.s2_data])

                print('----- forecast')
                forecast = self.sample_forecast_with_starter_seq(
                    [d[:][0] for d in self.len_data],
                    [d[:][0] for d in self.s1_data],
                    [d[:][0] for d in self.s2_data])
                print([g[0] + 1 for g in forecast])
                print([g[1] + 1 for g in forecast])
                print([g[2] + 1 for g in forecast])

        self.trained = True
        self.training = False

    def get_predictions_for_seq(self, len_seq, s1_seq, s2_seq):
        if not self.trained:
            raise Exception("RNN has not been trained")

        predictions = self.sess.run(self.generated_seq, feed_dict=dict(
            zip(self.single_len_sequence, [[s] for s in len_seq]) +
            zip(self.single_s1_sequence, [[s] for s in s1_seq]) +
            zip(self.single_s2_sequence, [[s] for s in s2_seq])))

        return predictions

    def sample_forecast_with_starter_seq(self, len_seq, s1_seq, s2_seq):
        if not (self.trained or self.training):
            raise Exception("RNN has not been trained")

        forecast = self.sess.run(self.forecast, feed_dict=dict(
            zip(self.single_len_sequence, [[s] for s in len_seq]) +
            zip(self.single_s1_sequence, [[s] for s in s1_seq]) +
            zip(self.single_s2_sequence, [[s] for s in s2_seq])))
        return forecast


if __name__ == '__main__':
    model = dist_rnn(periodic=True, noise=False)
    model.train()
