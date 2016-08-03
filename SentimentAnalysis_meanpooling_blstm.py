import os, cPickle, time

import numpy
import theano
import theano.tensor as T
import lasagne

from Res import *
from Utils import *


class MeanPoolingLayer(lasagne.layers.Layer):
    def __init__(self, incoming, mask_input, **kwargs):
        super(MeanPoolingLayer, self).__init__(incoming, **kwargs)
        self.mask = mask_input
    
    def get_output_for(self, input, **kwargs):
        mask_input = theano.tensor.switch(self.mask.dimshuffle(0, 1, 'x'), input, 0)
        output =  mask_input.sum(axis = 1) / self.mask.sum(axis = 1)[:, None]
        return theano.tensor.cast(output, dtype = theano.config.floatX)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])


def build_model(max_seq_length, input_var = None, mask_var = None):
    l_in = lasagne.layers.InputLayer((None, max_seq_length), input_var = input_var)

    l_mask = lasagne.layers.InputLayer((None, max_seq_length), input_var = mask_var)

    l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size = vocabulary_size, output_size = word_dim)

    l_lstm = lasagne.layers.LSTMLayer(l_emb, num_units = hiden_dim, grad_clipping = lstm_grad_clip,
                                        nonlinearity = lasagne.nonlinearities.tanh,
                                        mask_input = l_mask)


    l_lstm_back = lasagne.layers.LSTMLayer(l_emb, num_units = hiden_dim, grad_clipping = lstm_grad_clip,
                                        nonlinearity = lasagne.nonlinearities.tanh, backwards = True,
                                        mask_input = l_mask)

    l_concat = lasagne.layers.ConcatLayer([l_lstm, l_lstm_back], axis = 2)
    print lasagne.layers.get_output_shape(l_concat)

    l_meanpooling = MeanPoolingLayer(l_concat, mask_input = mask_var)

    l_drop = lasagne.layers.DropoutLayer(l_meanpooling, p = lstm_drop)

    l_out = lasagne.layers.DenseLayer(l_drop, num_units = num_class,
                                        nonlinearity = lasagne.nonlinearities.softmax)

    return l_out


def train():
    if not os.path.exists(train_dataset_path):
        generate_dataset()

    train_x, train_x_mask, train_y = cPickle.load(open(train_dataset_path, 'r'))
    valid_x, valid_x_mask, valid_y = cPickle.load(open(valid_dataset_path, 'r'))


    num_train_batchs = len(train_y) / batch_size
    num_valid_batchs = len(valid_y) / valid_batch_size

    print 't: %d, tb: %d, v: %d, vb: %d'%(len(train_y), num_train_batchs, len(valid_y), num_valid_batchs)


    shared_x_train, shared_y_train = shared_dataset(train_x, train_y)
    shared_mask = shared_data(train_x_mask, dtype = 'int8')

    shared_x_valid, shared_y_valid = shared_dataset(valid_x, valid_y)
    shared_valid_mask = shared_data(valid_x_mask, dtype = 'int8')


    index = T.lscalar('index')
    input_var = T.lmatrix('input')
    target_var = T.ivector('target')
    mask_var = T.bmatrix('mask')

    network = build_model(max_seq_length, input_var, mask_var)
    prediction = lasagne.layers.get_output(network)
    test_output = lasagne.layers.get_output(network, deterministic=True)

    test_acc =  T.mean( T.eq(T.argmax(test_output, axis = 1), target_var), dtype = theano.config.floatX)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()

    params = lasagne.layers.get_all_params(network, trainable = True)
    updates = lasagne.updates.adadelta(loss, params, learning_rate)

    train_fn = theano.function([index],
                            outputs = loss,
                            updates = updates,
                            givens={
                                    input_var: shared_x_train[index * batch_size: (index + 1) * batch_size],
                                    target_var: shared_y_train[index * batch_size: (index + 1) * batch_size],
                                    mask_var: shared_mask[index * batch_size: (index + 1) * batch_size],
                                    }
                            )

    valid_fn = theano.function([index],
                            outputs = test_acc,
                            givens={
                                    input_var: shared_x_valid[index * valid_batch_size: (index + 1) * valid_batch_size],
                                    target_var: shared_y_valid[index * valid_batch_size: (index + 1) * valid_batch_size],
                                    mask_var: shared_valid_mask[index * valid_batch_size: (index + 1) * valid_batch_size],
                                    }
                            )

    print 'compile over...'
    best_acc = 0.0
    for epoch in xrange(num_epoch):
        loss = 0.0
        acc = 0.0
        
        indices = range(0, num_train_batchs)
        numpy.random.shuffle(indices)
        start_time = time.time()

        for batch in indices:
            loss += train_fn(batch)

        valid_indices = range(0, num_valid_batchs)
        for batch in valid_indices:
            acc += valid_fn(batch)
        
        loss /= num_train_batchs
        acc /= num_valid_batchs

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epoch, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(loss))
        print("  valid accuracy:\t\t{:.2f} %\n".format(acc * 100))

        if best_acc < acc:
            best_acc = acc
            cPickle.dump((input_var, mask_var, network), open(meanpooling_blstm_path, 'w'))
            print 'save mblstm to %s, best valid accuracy: %.2f%%\n'%(meanpooling_blstm_path, best_acc * 100)


def test():
    test_x, test_x_mask, test_y = cPickle.load(open(test_dataset_path, 'r'))

    batch_size = test_batch_size
    num_test_batchs = len(test_y) / batch_size

    print 't: %d, tb: %d'%(len(test_y), num_test_batchs)

    index = T.lscalar('index')
    target_var = T.ivector('target')
    input_var, mask_var, network = cPickle.load(open(meanpooling_blstm_path, 'r'))
    prediction = lasagne.layers.get_output(network, deterministic = True)

    shared_x_test, shared_y_test = shared_dataset(test_x, test_y)
    shared_mask = shared_data(test_x_mask, dtype = 'int8')


    acc = T.mean( T.eq(T.argmax(prediction, axis = 1), target_var), dtype = theano.config.floatX)

    
    test_fn = theano.function([index],
                            outputs = acc,
                            givens={
                                    input_var: shared_x_test[index * batch_size: (index + 1) * batch_size],
                                    target_var: shared_y_test[index * batch_size: (index + 1) * batch_size],
                                    mask_var: shared_mask[index * batch_size: (index + 1) * batch_size],
                                    }
                            )

    indices = range(0, num_test_batchs)

    test_acc = 0.0
    for batch in indices:
        test_acc += test_fn(batch)

    test_acc /= num_test_batchs
    print 'test accuracy: %.2f%%\n'%(test_acc * 100)


if __name__ == '__main__':
    main(train, test)