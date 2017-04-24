import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np


class Model():
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size*2)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            cells.append(cell)

        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)



        self.input_left_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])

        self.input_right_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])

        self.targets = tf.placeholder(
            tf.int32, [args.batch_size,1])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        def init_weights(name,shape):
            return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=0.01))

        def init_biases(name,shape):
            return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=0.01))
        
        hidden_layer1_size=args.vocab_size*2
        hidden_layer2_size=args.rnn_size*2

        with tf.variable_scope('rnnlm'):
            w_h1 = init_weights("vanilla1_w", [args.rnn_size*2, hidden_layer1_size])
            b_h1 = init_biases("vanilla1_b", [hidden_layer1_size])
            w_h2 = init_weights("vanilla2_w", [args.rnn_size*2, hidden_layer2_size])
            b_h2 = init_biases("vanilla2_b", [hidden_layer2_size])
            softmax_w = init_weights("softmax_w", [args.rnn_size*2, args.vocab_size])
            softmax_b = init_weights("softmax_b", [args.vocab_size])


        left_embedding = tf.get_variable("left_embedding", [args.vocab_size, args.rnn_size])
        left_inputs = tf.nn.embedding_lookup(left_embedding, self.input_left_data)

        # dropout beta testing: double check which one should affect next line
        if training and args.output_keep_prob:
            left_inputs = tf.nn.dropout(left_inputs, args.output_keep_prob)


        left_inputs = tf.split(left_inputs, args.seq_length, 1)
        left_inputs = [tf.squeeze(input_, [1]) for input_ in left_inputs]


        right_embedding = tf.get_variable("right_embedding", [args.vocab_size, args.rnn_size])
        right_inputs = tf.nn.embedding_lookup(right_embedding, self.input_right_data)

        # dropout beta testing: double check which one should affect next line
        if training and args.output_keep_prob:
            right_inputs = tf.nn.dropout(right_inputs, args.output_keep_prob)


        right_inputs = tf.split(right_inputs, args.seq_length, 1)
        right_inputs = [tf.squeeze(input_, [1]) for input_ in right_inputs]

        inputs=[tf.reshape(tf.stack([left_input_,right_input_], axis=1),[args.batch_size,args.rnn_size*2]) for left_input_,right_input_ in zip(left_inputs,right_inputs)]


        def loop(prev, _):
            #p_h1 = tf.nn.relu(tf.matmul(prev, w_h1)+b_h1)
            #p_h2 = tf.nn.relu(tf.matmul(h1, w_h2)+b_h2)
            prev = tf.matmul(prev, softmax_w)+softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [args.batch_size,-1, args.rnn_size*2])
        output = tf.slice(output,[0, args.seq_length-1,0], [args.batch_size,1,args.rnn_size*2])
        output = tf.reshape(output, [-1, args.rnn_size*2])

        h1 = tf.nn.relu(tf.matmul(output, w_h1)+b_h1)
        h1 = tf.nn.dropout(h1, args.output_keep_prob)

        h2 = tf.nn.relu(tf.matmul(output, w_h2)+b_h2)
        h2 = tf.nn.dropout(h2, args.output_keep_prob)
        
        self.logits = tf.matmul(output, softmax_w)+softmax_b

        #self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size])])
        self.cost = tf.reduce_sum(loss) / args.batch_size
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # instrument tensorboard
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)

    def sample(self, sess, chars, vocab, left_xdata='The ', right_xdata='The ', num=200, sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for l, r in zip(left_xdata[0],right_xdata[0]):
            l_x = np.zeros((1, 1))
            l_x[0, 0] = l
            r_x = np.zeros((1, 1))
            r_x[0, 0] = r
            feed = {self.input_left_data: l_x,self.input_right_data: r_x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = str(" ")
        for l_x_row, r_x_row in zip(left_xdata,right_xdata):
            #x = np.zeros((1, 1))
            #x[0, 0] = vocab[char]
            feed = {self.input_left_data: l_x,self.input_right_data: r_x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = dict()

            for l, r in zip(l_x_row,r_x_row):
                l_x = np.zeros((1, 1))
                l_x[0, 0] = l
                r_x = np.zeros((1, 1))
                r_x[0, 0] = r
                feed = {self.input_left_data: l_x,self.input_right_data: r_x, self.initial_state: state}
                [probs, state] = sess.run([self.probs, self.final_state], feed)
                p['last'] = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p['last'])
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p['last'])
                else:
                    sample = np.argmax(p['last'])
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p['last'])

            pred = chars[sample]
            ret += pred
        return ret
##Host kudep@40.71.94.164
##  Hostname 40.71.94.164
##  RemoteForward 52698 127.0.0.1:52698
