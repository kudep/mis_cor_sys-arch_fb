from __future__ import print_function
import tensorflow as tf

import argparse
import os
from six.moves import cPickle

from model import Model

from six import text_type

import codecs
import numpy as np


def main():
    parser = argparse.ArgumentParser(
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                        help='data directory containing data_for_sample.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=500,
                        help='number of characters to sample')
    parser.add_argument('--seq_length', type=int, default=23,
                        help='RNN sequence length')
    parser.add_argument('--prime', type=text_type, default=u' ',
                        help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at '
                             'each timestep, 2 to sample on spaces')

    args = parser.parse_args()
    left_xdata,right_xdata = data_get(args)
    sample(args,left_xdata,right_xdata)


def data_get(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    data_file = os.path.join(args.data_dir, "data_for_sample.txt")

    seq_length=args.seq_length
    wind_len=seq_length*2+1

    with codecs.open(data_file, "r", encoding='utf-8') as f:
        data = f.read()

    meta_tensor = np.array(list(map(vocab.get, data)))
    if meta_tensor.size < wind_len:
            assert False, "Not enough data."

    cons_index=0
    cons=np.empty((meta_tensor.size, meta_tensor.size))
    indexs=np.split(range(meta_tensor.size),[meta_tensor.size//2+1])
    for j in range(len(indexs)):
        for i in indexs[len(indexs)-1-j]:
            con=np.concatenate([meta_tensor[i:],meta_tensor[0:i]])
            #print(con)
            #print(con.size)
            cons[cons_index]=con
            cons_index+=1

    bottom_index=meta_tensor.size//2-seq_length-1
    top_index=meta_tensor.size//2-seq_length
    mid_index=meta_tensor.size//2
    rl_xdata=np.arange(0)
    for i in range(meta_tensor.size):
        rl_xdata=np.concatenate([rl_xdata,cons[i][bottom_index:mid_index-1],cons[i][mid_index:meta_tensor.size-top_index]])
    left_xdata=np.copy(rl_xdata.reshape([-1,seq_length])[0::2].reshape([-1]))

    r_xdata=rl_xdata.reshape([-1,seq_length])[1::2].reshape([-1])

    right_xdata = np.copy(r_xdata[::-1].reshape([-1,seq_length])[::-1].reshape([-1]))

    #print(left_xdata[0])
    #print(right_xdata[0])
    #print("-------------------------------")
    #print(left_xdata[1])
    #print(right_xdata[2])
#
    #devocab = dict(zip(range(len(chars)), chars))
    #print(list(map(devocab.get,left_xdata.reshape([-1,seq_length])[10])))
    #print(list(map(devocab.get,right_xdata.reshape([-1,seq_length])[10])))

    return left_xdata.reshape([-1,seq_length]),right_xdata.reshape([-1,seq_length])





def sample(args,left_xdata,right_xdata):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(model.sample(sess, chars, vocab, left_xdata, right_xdata,
                               args.sample))

if __name__ == '__main__':
    main()
