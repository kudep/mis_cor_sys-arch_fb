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

        self.input_file = os.path.join(data_dir, "input.txt")
        self.vocab_file = os.path.join(data_dir, "vocab.pkl")

        self.left_xdata_file = os.path.join(data_dir, "left_xdata.npy")
        self.right_xdata_file = os.path.join(data_dir, "right_xdata.npy")
        self.ydata_file = os.path.join(data_dir, "ydata.npy")

        start_time=time.time()
        if not (os.path.exists(self.vocab_file) and os.path.exists(self.left_xdata_file) and os.path.exists(self.right_xdata_file) and os.path.exists(self.ydata_file)):
            print("reading text file")
            self.preprocess()
        else:
            print("loading preprocessed files")
            self.load_preprocessed()

        self.create_batches()
        self.reset_batch_pointer()
        print('preprocess time used = {0:.3f}'.format(time.time()-start_time))



    def preprocess(self):
        with codecs.open(self.input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)

        print()
        print()
        print('self.chars')
        print(self.chars)


        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(self.vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        #self.tensor = np.array(list(map(self.vocab.get, data)))
#
        #self.wind_len=self.seq_length*2+1
        #self.num_batches=(self.tensor.size)//(self.batch_size*self.wind_len)
        #self.tensor=self.tensor[:self.num_batches*self.batch_size*self.wind_len]
        #start
        self.meta_tensor = np.array(list(map(self.vocab.get, data)))

        self.wind_len=self.seq_length*2+1
        self.num_batches=(self.meta_tensor.size)//(self.batch_size*self.wind_len)

        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        print('Make data tensor')
        self.tensor=self.meta_tensor[:(self.num_batches*self.batch_size-1)*self.wind_len]

        num_parts=2
        parts=np.arange(1,self.wind_len//2).reshape(num_parts,-1) #Half all data
        for part_ind in range(num_parts):
            print('Start existing {0} part of data tensor'.format(part_ind))
            part_tensor=np.arange(0)
            for ind in tqdm(parts[part_ind]):
                part_tensor=np.concatenate([part_tensor,self.meta_tensor[ind:(self.num_batches*self.batch_size-1)*+ind]])
            self.tensor=np.concatenate([self.tensor,part_tensor])

        self.num_batches=(self.tensor.size)//(self.batch_size*self.wind_len)
        self.tensor=self.tensor[:(self.num_batches*self.batch_size)*self.wind_len]


        #end

        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.form_dataset()




    def extended_preprocess(self):
        with codecs.open(self.input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        
        frec_list, _ = zip(*count_pairs)
        accessable_vocab=['\n', ' ', '!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', '{', '|', '}', '~' '§', '©', '«', '®', '¯', '´', '·', '»','Ё', 'І', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё', '…', '–', '']
        self.chars=list()
        for i in frec_list:
            if(accessable_vocab.count(i)!=0):
                self.chars.append(i)
        self.chars.append('')


        self.vocab = dict(zip(self.chars, range(len(self.chars))))


        print()
        print()
        print('frec_list')
        print(frec_list)

        print()
        print()
        print('accessable_vocab')
        print(accessable_vocab)

        print()
        print()
        print('chars')
        print(self.chars)

        print()
        print()
        print('vocab')
        print(self.vocab)
        raise

        def mapping(char):
            return self.vocab.get(char, self.vocab.get(char,''))

        self.vocab_size = len(self.chars)
        with open(self.vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(mapping, data)))
        print(self.tensor[1:40])
        self.wind_len=self.seq_length*2+1
        self.num_batches=(self.tensor.size)//(self.batch_size*wind_len)
        self.tensor=self.tensor[:self.num_batches*self.batch_size*self.wind_len]

        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.form_dataset()



    def form_dataset(self,):
        #Form dataset to left_data, rught_data and target is ydata
        merg_data=self.tensor.reshape([-1,self.wind_len])
        num_parts=self.batch_size
        parts=np.arange(len(merg_data)).reshape(num_parts,-1)
        rl_xdata=np.arange(0)
        for part_ind in range(num_parts):
            print('Start existing {0} part of xdata'.format(part_ind))
            part_rl_xdata=np.arange(0)
            for i in tqdm(parts[part_ind]):
                part_rl_xdata=np.concatenate([part_rl_xdata,merg_data[i][:-self.seq_length-1],merg_data[i][self.seq_length+1:]])
            rl_xdata=np.concatenate([rl_xdata,part_rl_xdata])


        left_xdata=np.copy(rl_xdata.reshape([-1,self.seq_length])[0::2].reshape([-1]))

        r_xdata=rl_xdata.reshape([-1,self.seq_length])[1::2].reshape([-1])

        right_xdata = np.copy(r_xdata[::-1].reshape([-1,self.seq_length])[::-1].reshape([-1]))

        ydata = self.tensor[self.seq_length::self.wind_len]

        #self.test(merg_data,left_xdata,right_xdata,ydata)

        #Save dataset
        np.save(self.left_xdata_file, left_xdata)
        np.save(self.right_xdata_file, right_xdata)
        np.save(self.ydata_file, ydata)



    def test(self,merg_data,left_xdata,right_xdata,ydata):
        for i in range(10):
            devocab = dict(zip(range(len(self.chars)), self.chars))
            print("")
            print("")
            print(list(map(devocab.get, merg_data[i])))
            print(list(map(devocab.get,left_xdata.reshape([-1,self.seq_length])[i])))
            print(list(map(devocab.get,right_xdata.reshape([-1,self.seq_length])[i])))
            print(devocab.get(ydata[i]))



    def load_preprocessed(self,):
        with open(self.vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))



    def create_batches(self):

        left_xdata=np.load(self.left_xdata_file)
        right_xdata = np.load(self.right_xdata_file)
        ydata = np.load(self.ydata_file)

        self.num_batches=len(ydata)//self.batch_size
        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

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
