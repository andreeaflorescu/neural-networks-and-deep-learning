#!/usr/bin/env python
"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np
from mpi4py import MPI

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""

    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        # read data from gzip
        tr_d, va_d, te_d = load_data()

        # calculate the set length that each process has to calculate
        tr_set_length = np.ceil(len(tr_d[0])*1.0 / comm.size)
        va_set_length = np.ceil(len(va_d[0])*1.0 / comm.size)
        te_set_length = np.ceil(len(te_d[0])*1.0 / comm.size)

        # send only the corresponding data to each process
        for rank in range(1, comm.size):
            comm.send(tr_d[0][rank * tr_set_length:(rank + 1) * tr_set_length], dest=rank, tag=11)
            comm.send(tr_d[1][rank * tr_set_length:(rank + 1) * tr_set_length], dest=rank, tag=12)
            comm.send(va_d[0][rank * va_set_length:(rank + 1) * va_set_length], dest=rank, tag=13)
            comm.send(te_d[0][rank * te_set_length:(rank + 1) * te_set_length], dest=rank, tag=14)

        # process the corresponding part of the arrays

        all_training_inputs = [np.reshape(x, (784,1)) for x in tr_d[0][0:tr_set_length]]
        all_training_results = [vectorized_result(y) for y in tr_d[1][0:tr_set_length]]
        all_validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0][0:va_set_length]]
        all_test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0][0:te_set_length]]

        total_messages = (comm.size - 1) * 4
        while total_messages > 0:
            status = MPI.Status()
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            if tag == 11:
                all_training_inputs.extend(data)
                # TODO push back to training_inputs - might not work properly
            elif tag == 12:
                all_training_results.extend(data)
                # TODO push back to training_results - might not work properly
            elif tag == 13:
                all_validation_inputs.extend(data)
                # TODO push back to validation_inputs - might not work properly
            elif tag == 14:
                all_test_inputs.extend(data)
                # TODO push back to test_inputs - might not work properly
            total_messages -= 1

        training_data = zip(all_training_inputs, all_training_results)
        validation_data = zip(all_validation_inputs, va_d[1])
        test_data = zip(all_test_inputs, te_d[1])

    else:
        # receive arrays from master and process the corresponding part
        training_inputs=[]
        training_results=[]
        validation_inputs=[]
        test_inputs=[]

        training_data = None
        validation_data = None
        test_data = None

        for i in range(0,4):
            status = MPI.Status()
            data   = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == 11:
                training_inputs = [np.reshape(x, (784,1)) for x in data]
            elif tag == 12:
                training_results = [vectorized_result(y) for y in data]
            elif tag == 13:
                validation_inputs = [np.reshape(x, (784, 1)) for x in data]
            elif tag == 14:
                test_inputs = [np.reshape(x, (784, 1)) for x in data]

        comm.send(training_inputs, dest=0, tag=11)
        comm.send(training_results, dest=0, tag=12)
        comm.send(validation_inputs, dest=0, tag=13)
        comm.send(test_inputs, dest=0, tag=14)

    comm.bcast(training_data, root=0)
    comm.bcast(validation_data, root=0)
    comm.bcast(test_data, root=0)
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
