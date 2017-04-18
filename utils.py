import codecs
import os
import collections
from six.moves import cPickle
import numpy as np

import time
from tqdm import tqdm


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")
        start_time=time.time()
        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()
        print('preprocessed time used = {0:.3f}'.format(time.time()-start_time))

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):

        wind_len=self.seq_length*2+1
        self.num_batches=(self.tensor.size)//(self.batch_size*wind_len)
        self.tensor=self.tensor[:self.num_batches*self.batch_size*wind_len]
        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        merg_data=self.tensor.reshape([-1,wind_len])
        rl_xdata=np.arange(0)
        for i in tqdm(range(len(merg_data))):
            rl_xdata=np.concatenate([rl_xdata,merg_data[i][:-self.seq_length-1],merg_data[i][self.seq_length+1:]])

        left_xdata=np.copy(rl_xdata.reshape([-1,self.seq_length])[0::2].reshape([-1]))

        r_xdata=rl_xdata.reshape([-1,self.seq_length])[1::2].reshape([-1])

        right_xdata = np.copy(r_xdata[::-1].reshape([-1,self.seq_length])[::-1].reshape([-1]))



        ydata = self.tensor[self.seq_length::wind_len]

        self.left_x_batches = np.split(left_xdata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)
        self.right_x_batches = np.split(right_xdata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)

    def next_batch(self):
        l_x, r_x, y = self.left_x_batches[self.pointer], self.right_x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return l_x, r_x, y

    def reset_batch_pointer(self):
        self.pointer = 0
