#!/usr/bin/env python
import mnist_loader
import network
from mpi4py import MPI
#from timeit import Timer

#t = Timer("mnist_loader.load_data_wrapper()")
#comm = MPI.COMM_WORLD
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
net.SGD(training_data, 1, 10, 3.0, test_data=test_data)
