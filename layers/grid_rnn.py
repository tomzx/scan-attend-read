# import numpy as np
# import tensorflow as tf
from keras.engine import Layer
# from tensorflow.models.rnn import rnn_cell
# from tensorflow.models.rnn import seq2seq
# from tensorflow.contrib import grid_rnn


class GridRNN(Layer):
    def __init__(self, args, infer=False, **kwargs):
        super(GridRNN, self).__init__(**kwargs)
        # self.args = args
        # self.infer = infer
        # if infer:
        #     args.batch_size = 1
        #     args.seq_length = 1
        #
        # additional_cell_args = {}
        # if args.model == 'rnn':
        #     cell_fn = rnn_cell.BasicRNNCell
        # elif args.model == 'gru':
        #     cell_fn = rnn_cell.GRUCell
        # elif args.model == 'lstm':
        #     cell_fn = rnn_cell.BasicLSTMCell
        # elif args.model == 'gridlstm':
        #     cell_fn = grid_rnn.Grid2LSTMCell
        #     additional_cell_args.update({'use_peepholes': True, 'forget_bias': 1.0})
        # elif args.model == 'gridgru':
        #     cell_fn = grid_rnn.Grid2GRUCell
        # else:
        #     raise Exception("model type not supported: {}".format(args.model))
        #
        # cell = cell_fn(args.rnn_size, **additional_cell_args)
        #
        # self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

    def build(self, input_shape):
        # self.input_shape = input_shape
        pass

    def call(self, inputs, **kwargs):
        return inputs
        pass
        # self.input_data = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_length])
        # self.targets = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_length])
        # self.initial_state = self.cell.zero_state(self.args.batch_size, tf.float32)
        #
        # with tf.variable_scope('rnnlm'):
        #     softmax_w = tf.get_variable("softmax_w", [self.args.rnn_size, self.args.vocab_size])
        #     softmax_b = tf.get_variable("softmax_b", [self.args.vocab_size])
        #     with tf.device("/cpu:0"):
        #         embedding = tf.get_variable("embedding", [self.args.vocab_size, self.args.rnn_size])
        #         inputs = tf.split(1, self.args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
        #         inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        #
        # def loop(prev, _):
        #     prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
        #     prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
        #     return tf.nn.embedding_lookup(embedding, prev_symbol)
        #
        # outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, self.cell,
        #                                           loop_function=loop if self.infer else None, scope='rnnlm')
        # output = tf.reshape(tf.concat(1, outputs), [-1, self.args.rnn_size])
        # self.logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # self.probs = tf.nn.softmax(self.logits)
        # loss = seq2seq.sequence_loss_by_example([self.logits],
        #                                         [tf.reshape(self.targets, [-1])],
        #                                         [tf.ones([self.args.batch_size * self.args.seq_length])],
        #                                         self.args.vocab_size)
        # self.cost = tf.reduce_sum(loss) / self.args.batch_size / self.args.seq_length
        # self.final_state = last_state
        # self.lr = tf.Variable(0.0, trainable=False)
        # tvars = tf.trainable_variables()
        # grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
        #                                   self.args.grad_clip)
        # optimizer = tf.train.AdamOptimizer(self.lr)
        # self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def compute_output_shape(self, input_shape):
        pass