import os, time, cPickle

import numpy
import theano
import lasagne

from Res import *

def prepare_data(seqs, labels, maxlen = max_seq_length):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((n_samples, maxlen)).astype('int64')
    x_mask = numpy.zeros((n_samples, maxlen)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.

    y = numpy.array(labels, dtype = 'float32')

    return x, x_mask, y


def load_data(path=dataset_path, n_words=vocabulary_size, valid_portion=0.1, maxlen=max_seq_length,
              sort_by_len=True):
    '''Loads the dataset

    :type path: String
    :param path: The path to the dataset (here IMDB)
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknow (1).
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.
    '''

    f = open(path, 'rb')

    train_set = cPickle.load(f)
    test_set = cPickle.load(f)
    f.close()
    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    return train, valid, test


def generate_dataset(dataset_path = dataset_path):
    train_set, vaild_set, test_set = load_data()

    
    def dataset_format(data_set):
        x, y = data_set
        return prepare_data(x, y, max_seq_length)

    train_x, train_x_mask, train_y = dataset_format(train_set)
    valid_x, valid_x_mask, valid_y = dataset_format(vaild_set)
    test_x, test_x_mask, test_y = dataset_format(test_set)

    cPickle.dump((train_x, train_x_mask, train_y), open(train_dataset_path, 'w'))
    cPickle.dump((valid_x, valid_x_mask, valid_y), open(valid_dataset_path, 'w'))
    cPickle.dump((test_x, test_x_mask, test_y), open(test_dataset_path, 'w'))



def shared_dataset(values, labels, borrow = True):
    shared_values = theano.shared(numpy.asarray(values, 
                                                dtype = theano.config.floatX
                                                ),
                                    borrow = borrow
                                    )

    shared_labels = theano.shared(numpy.asarray(labels, 
                                                dtype = theano.config.floatX
                                                ),
                                    borrow = borrow
                                    )

    return theano.tensor.cast(shared_values, 'int64'), theano.tensor.cast(shared_labels, 'int32')


def shared_data(values, borrow = True, dtype = theano.config.floatX):
    data =  theano.shared(numpy.asarray(values, 
                                                dtype = theano.config.floatX
                                                ),
                                    borrow = borrow
                                    )

    if dtype != theano.config.floatX:
        return theano.tensor.cast(data, dtype)
    else:
        return data


def main(train_func = None, test_func = None):
    if train_func:
        #'''
        print 'start train...'
        start_time = time.time()
        train_func()
        end_time = time.time()
        mins = (end_time - start_time) / 60
        secs = (end_time - start_time) % 60
        print 'Train cost:%2dm %2ds\n'%(mins, secs)
        #'''
    
    if test_func:
        print 'start test...'
        start_time = time.time()
        test_func()
        end_time = time.time()
        mins = (end_time - start_time) / 60
        secs = (end_time - start_time) % 60
        print 'Test cost:%2dm %2ds'%(mins, secs)
        #'''

    print 'end...'

