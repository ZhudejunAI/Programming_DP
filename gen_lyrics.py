#coding=utf-8
import os
import sys
import time
import codecs
import numpy as np
import tensorflow as tf
from numpy import shape
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib import legacy_seq2seq
from tensorflow.contrib import rnn
from tensorflow.contrib.deprecated import scalar_summary, histogram_summary, merge_all_summaries


class HParam():
    def __init__(self):
        pass

    batch_size = 32
    n_epoch = 100
    learning_rate = 0.01
    decay_steps = 1000
    decay_rate = 0.9
    grad_clip = 5
    state_size = 100
    num_layer = 3
    seq_length = 20
    log_dir = './logs'
    metadata = 'metadata.tsv'
    gen_num = 500


class DataGenerator():
    def __init__(self, datafile, args):
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length
        self.data = ''
        f = codecs.open(datafile, 'r', encoding='utf-8')
        self.data = f.read()
        f.close()

        self.total_len = len(self.data)
        self.words = list(set(self.data))
        self.words.sort()
        self.vocab_size = len(self.words)
        print("Vocab_size:", self.vocab_size)
        self.char2id_dict = {w: i for i, w in enumerate(self.words)}
        self.id2char_dict = {i: w for i, w in enumerate(self.words)}
        self._pointer = 0
        self.save_metadata = args.metadata

    def char2id(self, c):
        return self.char2id_dict[c]

    def id2char(self, id):
        return self.id2char_dict[id]

    def save_metadata(self, filename):
        with open(filename) as f:
            f.write('id\tchar\n')
            for i in xrange(self.vocab_size):
                c = self.id2char_dict[i]
                f.write('{}\t{}\n'.format(i, c))

    def next_batch(self):
        x_batches = []
        y_batches = []
        for i in range(self.batch_size):
            if self._pointer + self.seq_length + 1 > self.total_len:
                self._pointer = 0

            bx = self.data[self._pointer:self._pointer + self.seq_length]
            by = self.data[self._pointer + 1:self._pointer + self.seq_length + 1]
            self._pointer += self.seq_length
            bx = [self.char2id_dict[c] for c in bx]
            by = [self.char2id_dict[c] for c in by]
            x_batches.append(bx)
            y_batches.append(by)

        return x_batches, y_batches


class Model():
    def __init__(self, args, data):
        with tf.name_scope("inputs"):
            self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
            self.target_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])

        with tf.name_scope("model"):
            self.cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(args.state_size) for _ in range(args.num_layer)])
            self.initial_state = self.cell.zero_state(args.batch_size, tf.float32)
            with tf.variable_scope("rnnlm"):
                w = tf.get_variable('softmax_w', [args.state_size, data.vocab_size])
                b = tf.get_variable('softmax_b', [data.vocab_size])
                with tf.device('/cpu:0'):
                    embedding = tf.get_variable('embedding', [data.vocab_size, args.state_size])
                    inputs = tf.nn.embedding_lookup(embedding, self.input_data)
                outputs, last_state = tf.nn.dynamic_rnn(self.cell, inputs, initial_state=self.initial_state)

        with tf.name_scope("loss"):
            output = tf.reshape(outputs, [-1, args.state_size])
            self.logits = tf.matmul(output, w) + b
            self.probs = tf.nn.softmax(self.logits)
            self.last_state = last_state

            targets = tf.reshape(self.target_data, [-1])
            loss = legacy_seq2seq.sequence_loss_by_example([self.logits], [targets],
                                                       [tf.ones_like(targets, dtype=tf.float32)])
            self.cost = tf.reduce_sum(loss) / args.batch_size
            scalar_summary('loss', self.cost)

        with tf.name_scope('optimize'):
            self.lr = tf.placeholder(tf.float32, [])
            scalar_summary('learning_rate', self.lr)

            optimizer = tf.train.AdamOptimizer(self.lr)
            train_vars = tf.trainable_variables()
            grads = tf.gradients(self.cost, train_vars)
            for g in grads:
                histogram_summary(g.name, g)
            grads, _ = tf.clip_by_global_norm(grads, args.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(grads, train_vars))
            self.merged_op = merge_all_summaries()


def train(data, model, args):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)

        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'rnnlm/embedding:0'
        embed.metadata_path = args.metadata
        projector.visualize_embeddings(writer, config)

        max_iter = args.n_epoch * (data.total_len // args.seq_length) // args.batch_size
        for i in range(max_iter):
            learning_rate = args.learning_rate * (args.decay_rate ** (i // args.decay_steps))
            x_batch, y_batch = data.next_batch()
            feed_dict = {model.input_data: x_batch, model.target_data: y_batch, model.lr: learning_rate}
            train_loss, summary, _, _ = sess.run([model.cost, model.merged_op, model.last_state, model.train_op],
                                                 feed_dict)
            if i % 10 == 0:
                writer.add_summary(summary, global_step=i)
                print('Step:{}/{}, training_loss:{:4f}'.format(i, max_iter, train_loss))
            if i % 2000 == 0 or (i + 1) == max_iter:
                saver.save(sess, os.path.join(args.log_dir, 'lyrics_model.ckpt'), global_step=i)


def sample(data, model, args):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(args.log_dir)
        print(ckpt)
        saver.restore(sess, ckpt)

        prime = u'你要离开我知道很简单'
        state = sess.run(model.cell.zero_state(32, tf.float32))
        for word in prime[:-1]:
            x = np.zeros((32, 20))
            x[0, 0] = data.char2id(word)
            feed = {model.input_data: x, model.initial_state: state}
            state = sess.run(model.last_state, feed)
        word = prime[-1]
        lyrics = prime
        for i in range(args.gen_num):
            x = np.zeros([32, 20])
            x[0, 0] = data.char2id(word)
            feed_dict = {model.input_data: x, model.initial_state: state}
            probs, state = sess.run([model.probs, model.last_state], feed_dict)
            p = probs[0]
            word = data.id2char(np.argmax(p))
            print(word,)
            sys.stdout.flush()
            time.sleep(0.05)
            lyrics += word
        return lyrics


def main():
    args = HParam()
    data = DataGenerator('JayLyrics.txt', args)
    model = Model(args, data)
    msg = sample(data, model, args)
    print(msg)

if __name__ == '__main__':
    main()
